/*
author: samylee
github: https://github.com/samylee
date: 08/19/2021
*/

#include "JDETracker.h"

int main()
{
	string model_path = "model/jde_576x320_torch14_gpu.pt";
	JDETracker tracker(model_path);

	string video_path = "video/AVG-TownCentre.mp4";
	tracker.update(video_path);

	return 0;
}
