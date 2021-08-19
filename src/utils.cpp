/*
author: samylee
github: https://github.com/samylee
date: 08/19/2021
*/

#include "JDETracker.h"
#include "lapjv.h"

Size JDETracker::get_size(int vw, int vh, int dw, int dh)
{
	Size size;
	float wa = float(dw) / vw;
	float ha = float(dh) / vh;
	float a = min(wa, ha);
	size.width = int(vw*a);
	size.height = int(vh*a);

	return size;
}

Mat JDETracker::letterbox(Mat img, int height, int width)
{
	Size shape = Size(img.cols, img.rows);
	float ratio = min(float(height) / shape.height, float(width) / shape.width);
	Size new_shape = Size(round(shape.width*ratio), round(shape.height*ratio));
	float dw = float(width - new_shape.width) / 2;
	float dh = float(height - new_shape.height) / 2;
	int top = round(dh - 0.1);
	int bottom = round(dh + 0.1);
	int left = round(dw - 0.1);
	int right = round(dw + 0.1);

	resize(img, img, new_shape, INTER_AREA);
	copyMakeBorder(img, img, top, bottom, left, right, BORDER_CONSTANT, Scalar(127.5, 127.5, 127.5));
	return img;
}

torch::Tensor JDETracker::nms(const torch::Tensor& boxes, const torch::Tensor& scores, float overlap)
{
	int count = 0;
	int top_k = 200;
	torch::Tensor keep = torch::empty({ scores.size(0) }).to(torch::kLong).to(scores.device()).fill_(-1);

	torch::Tensor x1 = boxes.select(1, 0).clone();
	torch::Tensor y1 = boxes.select(1, 1).clone();
	torch::Tensor x2 = boxes.select(1, 2).clone();
	torch::Tensor y2 = boxes.select(1, 3).clone();
	torch::Tensor area = (x2 - x1)*(y2 - y1);

	std::tuple<torch::Tensor, torch::Tensor> sort_ret = torch::sort(scores.unsqueeze(1), 0, 0);
	torch::Tensor v = std::get<0>(sort_ret).squeeze(1).to(scores.device());
	torch::Tensor idx = std::get<1>(sort_ret).squeeze(1).to(scores.device());

	int num_ = idx.size(0);
	if (num_ > top_k) //python:idx = idx[-top_k:]
	{
		idx = idx.slice(0, num_ - top_k, num_).clone();
	}
	torch::Tensor xx1, yy1, xx2, yy2, w, h;
	while (idx.numel() > 0)
	{
		auto i = idx[-1];
		keep[count] = i;
		count += 1;
		if (1 == idx.size(0))
		{
			break;
		}
		idx = idx.slice(0, 0, idx.size(0) - 1).clone();

		xx1 = x1.index_select(0, idx);
		yy1 = y1.index_select(0, idx);
		xx2 = x2.index_select(0, idx);
		yy2 = y2.index_select(0, idx);

		xx1 = xx1.clamp(x1[i].item().toFloat(), INT_MAX*1.0);
		yy1 = yy1.clamp(y1[i].item().toFloat(), INT_MAX*1.0);
		xx2 = xx2.clamp(INT_MIN*1.0, x2[i].item().toFloat());
		yy2 = yy2.clamp(INT_MIN*1.0, y2[i].item().toFloat());

		w = xx2 - xx1;
		h = yy2 - yy1;

		w = w.clamp(0, INT_MAX);
		h = h.clamp(0, INT_MAX);

		torch::Tensor inter = w * h;
		torch::Tensor rem_areas = area.index_select(0, idx);

		torch::Tensor union_ = (rem_areas - inter) + area[i];
		torch::Tensor Iou = inter * 1.0 / union_;
		torch::Tensor index_small = Iou < overlap;
		auto mask_idx = torch::nonzero(index_small).squeeze();
		idx = idx.index_select(0, mask_idx);//pthon: idx = idx[IoU.le(overlap)]
	}
	return keep.index_select(0, torch::nonzero(keep >= 0).squeeze());
}

torch::Tensor JDETracker::xywh2xyxy(torch::Tensor x)
{
	auto y = torch::zeros_like(x);
	y.slice(1, 0, 1) = x.slice(1, 0, 1) - x.slice(1, 2, 3) / 2;
	y.slice(1, 1, 2) = x.slice(1, 1, 2) - x.slice(1, 3, 4) / 2;
	y.slice(1, 2, 3) = x.slice(1, 0, 1) + x.slice(1, 2, 3) / 2;
	y.slice(1, 3, 4) = x.slice(1, 1, 2) + x.slice(1, 3, 4) / 2;

	return y;
}

torch::Tensor JDETracker::non_max_suppression(torch::Tensor prediction)
{
	prediction.slice(1, 0, 4) = xywh2xyxy(prediction.slice(1, 0, 4));
	torch::Tensor nms_indices = nms(prediction.slice(1, 0, 4), prediction.select(1, 4), nms_thresh);

	return prediction.index_select(0, nms_indices);
}

void JDETracker::scale_coords(torch::Tensor &coords, Size img_size, Size img0_shape)
{
	float gain_w = float(img_size.width) / img0_shape.width;
	float gain_h = float(img_size.height) / img0_shape.height;
	float gain = min(gain_w, gain_h);
	float pad_x = (img_size.width - img0_shape.width*gain) / 2;
	float pad_y = (img_size.height - img0_shape.height*gain) / 2;
	coords.select(1, 0) -= pad_x;
	coords.select(1, 1) -= pad_y;
	coords.select(1, 2) -= pad_x;
	coords.select(1, 3) -= pad_y;
	coords /= gain;
	coords = torch::clamp(coords, 0);
}

vector<STrack*> JDETracker::joint_stracks(vector<STrack*> &tlista, vector<STrack> &tlistb)
{
	map<int, int> exists;
	vector<STrack*> res;
	for (int i = 0; i < tlista.size(); i++)
	{
		exists.insert(pair<int, int>(tlista[i]->track_id, 1));
		res.push_back(tlista[i]);
	}
	for (int i = 0; i < tlistb.size(); i++)
	{
		int tid = tlistb[i].track_id;
		if (!exists[tid] || exists.count(tid) == 0)
		{
			exists[tid] = 1;
			res.push_back(&tlistb[i]);
		}
	}
	return res;
}

vector<STrack> JDETracker::joint_stracks(vector<STrack> &tlista, vector<STrack> &tlistb)
{
	map<int, int> exists;
	vector<STrack> res;
	for (int i = 0; i < tlista.size(); i++)
	{
		exists.insert(pair<int, int>(tlista[i].track_id, 1));
		res.push_back(tlista[i]);
	}
	for (int i = 0; i < tlistb.size(); i++)
	{
		int tid = tlistb[i].track_id;
		if (!exists[tid] || exists.count(tid) == 0)
		{
			exists[tid] = 1;
			res.push_back(tlistb[i]);
		}
	}
	return res;
}

vector<STrack> JDETracker::sub_stracks(vector<STrack> &tlista, vector<STrack> &tlistb)
{
	map<int, STrack> stracks;
	for (int i = 0; i < tlista.size(); i++)
	{
		stracks.insert(pair<int, STrack>(tlista[i].track_id, tlista[i]));
	}
	for (int i = 0; i < tlistb.size(); i++)
	{
		int tid = tlistb[i].track_id;
		if (stracks.count(tid) != 0)
		{
			stracks.erase(tid);
		}
	}

	vector<STrack> res;
	std::map<int, STrack>::iterator  it;
	for (it = stracks.begin(); it != stracks.end(); ++it)
	{
		res.push_back(it->second);
	}

	return res;
}

void JDETracker::remove_duplicate_stracks(vector<STrack> &resa, vector<STrack> &resb, vector<STrack> &stracksa, vector<STrack> &stracksb)
{
	vector<vector<float> > pdist = iou_distance(stracksa, stracksb);
	vector<pair<int, int> > pairs;
	for (int i = 0; i < pdist.size(); i++)
	{
		for (int j = 0; j < pdist[i].size(); j++)
		{
			if (pdist[i][j] < 0.15)
			{
				pairs.push_back(pair<int, int>(i, j));
			}
		}
	}

	vector<int> dupa, dupb;
	for (int i = 0; i < pairs.size(); i++)
	{
		int timep = stracksa[pairs[i].first].frame_id - stracksa[pairs[i].first].start_frame;
		int timeq = stracksb[pairs[i].second].frame_id - stracksb[pairs[i].second].start_frame;
		if (timep > timeq)
			dupb.push_back(pairs[i].second);
		else
			dupa.push_back(pairs[i].first);
	}

	for (int i = 0; i < stracksa.size(); i++)
	{
		vector<int>::iterator iter = find(dupa.begin(), dupa.end(), i);
		if (iter == dupa.end())
		{
			resa.push_back(stracksa[i]);
		}
	}

	for (int i = 0; i < stracksb.size(); i++)
	{
		vector<int>::iterator iter = find(dupb.begin(), dupb.end(), i);
		if (iter == dupb.end())
		{
			resb.push_back(stracksb[i]);
		}
	}
}

void JDETracker::embedding_distance(vector<STrack*> &tracks, vector<STrack> &detections,
	vector<vector<float> > &cost_matrix, int *cost_matrix_size, int *cost_matrix_size_size)
{
	if (tracks.size() * detections.size() == 0)
	{
		*cost_matrix_size = tracks.size();
		*cost_matrix_size_size = detections.size();
		return;
	}

	for (int i = 0; i < tracks.size(); i++)
	{
		vector<float> cost_matrix_tmp;
		vector<float> track_feature = tracks[i]->smooth_feat;
		for (int j = 0; j < detections.size(); j++)
		{
			vector<float> det_feature = detections[j].curr_feat;
			float feat_square = 0.0;
			for (int k = 0; k < det_feature.size(); k++)
			{
				feat_square += (track_feature[k] - det_feature[k])*(track_feature[k] - det_feature[k]);
			}
			cost_matrix_tmp.push_back(sqrt(feat_square));
		}
		cost_matrix.push_back(cost_matrix_tmp);
	}
	*cost_matrix_size = tracks.size();
	*cost_matrix_size_size = detections.size();
}

void JDETracker::fuse_motion(vector<vector<float> > &cost_matrix, vector<STrack*> &tracks, vector<STrack> &detections,
	bool only_position, float lambda_)
{
	if (cost_matrix.size() == 0)
		return;

	int gating_dim = 0;
	if (only_position)
		gating_dim = 2;
	else
		gating_dim = 4;

	float gating_threshold = this->kalman_filter.chi2inv95[gating_dim];

	vector<DETECTBOX> measurements;
	for (int i = 0; i < detections.size(); i++)
	{
		DETECTBOX measurement;
		vector<float> tlwh_ = detections[i].to_xyah();
		measurement[0] = tlwh_[0];
		measurement[1] = tlwh_[1];
		measurement[2] = tlwh_[2];
		measurement[3] = tlwh_[3];
		measurements.push_back(measurement);
	}

	for (int i = 0; i < tracks.size(); i++)
	{
		Eigen::Matrix<float, 1, -1> gating_distance = kalman_filter.gating_distance(
			tracks[i]->mean, tracks[i]->covariance, measurements, only_position);

		for (int j = 0; j < cost_matrix[i].size(); j++)
		{
			if (gating_distance[j] > gating_threshold)
			{
				cost_matrix[i][j] = FLT_MAX;
			}
			cost_matrix[i][j] = lambda_ * cost_matrix[i][j] + (1 - lambda_)*gating_distance[j];
		}
	}
}

void JDETracker::linear_assignment(vector<vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
	vector<vector<int> > &matches, vector<int> &unmatched_a, vector<int> &unmatched_b)
{
	if (cost_matrix.size() == 0)
	{
		for (int i = 0; i < cost_matrix_size; i++)
		{
			unmatched_a.push_back(i);
		}
		for (int i = 0; i < cost_matrix_size_size; i++)
		{
			unmatched_b.push_back(i);
		}
		return;
	}

	vector<int> rowsol; vector<int> colsol;
	float c = lapjv(cost_matrix, rowsol, colsol, true, 0.7);
	for (int i = 0; i < rowsol.size(); i++)
	{
		if (rowsol[i] >= 0)
		{
			vector<int> match;
			match.push_back(i);
			match.push_back(rowsol[i]);
			matches.push_back(match);
		}
		else
		{
			unmatched_a.push_back(i);
		}
	}

	for (int i = 0; i < colsol.size(); i++)
	{
		if (colsol[i] < 0)
		{
			unmatched_b.push_back(i);
		}
	}
}

vector<vector<float> > JDETracker::ious(vector<vector<float> > &atlbrs, vector<vector<float> > &btlbrs)
{
	vector<vector<float> > ious;
	if (atlbrs.size()*btlbrs.size() == 0)
		return ious;

	ious.resize(atlbrs.size());
	for (int i = 0; i < ious.size(); i++)
	{
		ious[i].resize(btlbrs.size());
	}

	//bbox_ious
	for (int k = 0; k < btlbrs.size(); k++)
	{
		vector<float> ious_tmp;
		float box_area = (btlbrs[k][2] - btlbrs[k][0] + 1)*(btlbrs[k][3] - btlbrs[k][1] + 1);
		for (int n = 0; n < atlbrs.size(); n++)
		{
			float iw = min(atlbrs[n][2], btlbrs[k][2]) - max(atlbrs[n][0], btlbrs[k][0]) + 1;
			if (iw > 0)
			{
				float ih = min(atlbrs[n][3], btlbrs[k][3]) - max(atlbrs[n][1], btlbrs[k][1]) + 1;
				if(ih > 0)
				{
					float ua = (atlbrs[n][2] - atlbrs[n][0] + 1)*(atlbrs[n][3] - atlbrs[n][1] + 1) + box_area - iw * ih;
					ious[n][k] = iw * ih / ua;
				}
				else
				{
					ious[n][k] = 0.0;
				}
			}
			else
			{
				ious[n][k] = 0.0;
			}
		}
	}

	return ious;
}

vector<vector<float> > JDETracker::iou_distance(vector<STrack*> &atracks, vector<STrack> &btracks, int &dist_size, int &dist_size_size)
{
	vector<vector<float> > atlbrs, btlbrs;
	for (int i = 0; i < atracks.size(); i++)
	{
		atlbrs.push_back(atracks[i]->tlbr);
	}
	for (int i = 0; i < btracks.size(); i++)
	{
		btlbrs.push_back(btracks[i].tlbr);
	}

	dist_size = atracks.size();
	dist_size_size = btracks.size();

	vector<vector<float> > _ious = ious(atlbrs, btlbrs);
	vector<vector<float> > cost_matrix;
	for (int i = 0; i < _ious.size();i++)
	{
		vector<float> _iou;
		for (int j = 0; j < _ious[i].size(); j++)
		{
			_iou.push_back(1 - _ious[i][j]);
		}
		cost_matrix.push_back(_iou);
	}

	return cost_matrix;
}

vector<vector<float> > JDETracker::iou_distance(vector<STrack> &atracks, vector<STrack> &btracks)
{
	vector<vector<float> > atlbrs, btlbrs;
	for (int i = 0; i < atracks.size(); i++)
	{
		atlbrs.push_back(atracks[i].tlbr);
	}
	for (int i = 0; i < btracks.size(); i++)
	{
		btlbrs.push_back(btracks[i].tlbr);
	}

	vector<vector<float> > _ious = ious(atlbrs, btlbrs);
	vector<vector<float> > cost_matrix;
	for (int i = 0; i < _ious.size(); i++)
	{
		vector<float> _iou;
		for (int j = 0; j < _ious[i].size(); j++)
		{
			_iou.push_back(1 - _ious[i][j]);
		}
		cost_matrix.push_back(_iou);
	}

	return cost_matrix;
}

double JDETracker::lapjv(const vector<vector<float> > &cost, vector<int> &rowsol, vector<int> &colsol,
	bool extend_cost, float cost_limit, bool return_cost)
{
	vector<vector<float> > cost_c;
	cost_c.assign(cost.begin(), cost.end());

	vector<vector<float> > cost_c_extended;

	int n_rows = cost.size();
	int n_cols = cost[0].size();
	rowsol.resize(n_rows);
	colsol.resize(n_cols);

	int n = 0;
	if (n_rows == n_cols)
	{
		n = n_rows;
	}
	else
	{
		if (!extend_cost)
		{
			cout << "set extend_cost=True" << endl;
			system("pause");
			exit(0);
		}
	}
		
	if (extend_cost || cost_limit < LONG_MAX)
	{
		n = n_rows + n_cols;
		cost_c_extended.resize(n);
		for (int i = 0; i < cost_c_extended.size(); i++)
			cost_c_extended[i].resize(n);

		if (cost_limit < LONG_MAX)
		{
			for (int i = 0; i < cost_c_extended.size(); i++)
			{
				for (int j = 0; j < cost_c_extended[i].size(); j++)
				{
					cost_c_extended[i][j] = cost_limit / 2.0;
				}
			}
		}
		else
		{
			float cost_max = -1;
			for (int i = 0; i < cost_c.size(); i++)
			{
				for (int j = 0; j < cost_c[i].size(); j++)
				{
					if (cost_c[i][j] > cost_max)
						cost_max = cost_c[i][j];
				}
			}
			for (int i = 0; i < cost_c_extended.size(); i++)
			{
				for (int j = 0; j < cost_c_extended[i].size(); j++)
				{
					cost_c_extended[i][j] = cost_max + 1;
				}
			}
		}

		for (int i = n_rows; i < cost_c_extended.size(); i++)
		{
			for (int j = n_cols; j < cost_c_extended[i].size(); j++)
			{
				cost_c_extended[i][j] = 0;
			}
		}
		for (int i = 0; i < n_rows; i++)
		{
			for (int j = 0; j < n_cols; j++)
			{
				cost_c_extended[i][j] = cost_c[i][j];
			}
		}

		cost_c.clear();
		cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
	}

	double **cost_ptr;
	cost_ptr = new double *[sizeof(double *) * n];
	for (int i = 0; i < n; i++)
		cost_ptr[i] = new double[sizeof(double) * n];

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cost_ptr[i][j] = cost_c[i][j];
		}
	}

	int* x_c = new int[sizeof(int) * n];
	int *y_c = new int[sizeof(int) * n];

	int ret = lapjv_internal(n, cost_ptr, x_c, y_c);
	if (ret != 0)
	{
		cout << "Calculate Wrong!" << endl;
		system("pause");
		exit(0);
	}

	double opt = 0.0;

	if (n != n_rows)
	{
		for (int i = 0; i < n; i++)
		{
			if (x_c[i] >= n_cols)
				x_c[i] = -1;
			if (y_c[i] >= n_rows)
				y_c[i] = -1;
		}
		for (int i = 0; i < n_rows; i++)
		{
			rowsol[i] = x_c[i];
		}
		for (int i = 0; i < n_cols; i++)
		{
			colsol[i] = y_c[i];
		}

		if (return_cost)
		{
			for (int i = 0; i < rowsol.size(); i++)
			{
				if (rowsol[i] != -1)
				{
					//cout << i << "\t" << rowsol[i] << "\t" << cost_ptr[i][rowsol[i]] << endl;
					opt += cost_ptr[i][rowsol[i]];
				}
			}
		}
	}
	else if (return_cost)
	{
		for (int i = 0; i < rowsol.size(); i++)
		{
			opt += cost_ptr[i][rowsol[i]];
		}
	}

	for (int i = 0; i < n; i++)
	{
		delete[]cost_ptr[i];
	}
	delete[]cost_ptr;
	delete[]x_c;
	delete[]y_c;

	return opt;
}

Scalar JDETracker::get_color(int idx)
{
	idx += 3;
	return Scalar(37 * idx % 255, 17 * idx % 255, 29 * idx % 255);
}