
double SmoothStep(double xMin, double xMax, double x);

double DetailRemap(const double alpha, double delta, double sigmaR)
{
	double fraction = delta / sigmaR; // 0.0 / 1.0
	double polynomial = pow(fraction, alpha); // pow(0.0, 1.0)
	if (alpha < 1.0) // always false because if (1.0 < 1.0)
	{
		const double kNoiseLevel = 0.01;
		double blend = SmoothStep(kNoiseLevel, 2.0 * kNoiseLevel, fraction * sigmaR);
		polynomial = blend * polynomial + (1.0 - blend) * fraction;
	}
	return polynomial; // 0.0
}

double EdgeRemap(const double beta, double delta)
{
	return beta * delta;
}

double SmoothStep(double xMin, double xMax, double x)
{
	double y = (x - xMin) / (xMax - xMin);
	y = fmax(0.0, fmin(1.0, y));
	return pow(y, 2.0) * pow(y - 2.0, 2.0);
}

__kernel void GetValueOfB(
	__global const float* memMatG0Data,
	const int G0Width,
	const int G0Height,
	const int layerIndex,
	const int halfWidthFilter,
	__global const float* memCoeffGRData,
	const int matCoeffGRWidth,
	__global const float* memCoeffGUpRData,
	const int matCoeffGUpRWidth,
	__global const float* memCoeffGCData,
	const int matCoeffGCWidth,
	__global const float* memCoeffGUpCData,
	const int matCoeffGUpCWidth,
	__global float* memMatB,
	const int BWidth,
	const int BHeight,
	__global const float* memMatG,
	const float sigmaR,
	const float alpha,
	const float beta)
{
	uint xB = get_global_id(0);
	uint yB = get_global_id(1);

	if (xB >= BWidth)
		return;
	if (yB >= BHeight)
		return;

	printf("(%d, %d)\n", xB, yB);

	int yG0 = yB << layerIndex;
	const int nROITop = max(yG0 - halfWidthFilter, 0);
	const int nROIBottom = min(yG0 + halfWidthFilter, G0Height - 1);
	const int nROIHeight = nROIBottom - nROITop + 1;

	int xG0 = xB << layerIndex;
	const int nROILeft = max(xG0 - halfWidthFilter, 0);
	const int nROIRight = min(xG0 + halfWidthFilter, G0Width - 1);
	const int nROIWidth = nROIRight - nROILeft + 1;

	__global const float* pnG0Local = (__global const float*)(memMatG0Data + nROITop * G0Width + nROILeft);

	memCoeffGRData += matCoeffGRWidth * yB;
	memCoeffGUpRData += matCoeffGUpRWidth * yB;

	memCoeffGCData += matCoeffGCWidth * xB;
	memCoeffGUpCData += matCoeffGUpCWidth * xB;

	const double reference = memMatG[yB * BWidth + xB]; // should be 0.0

	double dSum = 0.0;
	for (int nY = 0; nY < nROIHeight; nY++)
	{
		for (int nX = 0; nX < nROIWidth; nX++)
		{
			double value = pnG0Local[nY * G0Width + nX]; // should be 0.0
			double delta = fabs(value - reference); // should be 0.0
			double sign = value < reference ? -1. : 1.; // should be 1.0
			double output = 0.;
			if (delta < sigmaR) // should be true because (0.0 < 1.0) 
				// should be 0.0 = 0.0 + 1.0 * 1.0 * DetailRemap(1.0, 0.0, 1.0);
				// should be 0.0 = 0.0 + 1.0 * 1.0 * 0.0
				output = reference + sign * sigmaR * DetailRemap(alpha, delta, sigmaR);
			else
				// next line should not be called
				output = reference + sign * (EdgeRemap(beta, delta - sigmaR) + sigmaR);
			dSum += (memCoeffGCData[nX] * memCoeffGRData[nY] - memCoeffGUpCData[nX] * memCoeffGUpRData[nY]) * output;
		}
	}

	printf("(%d, %d, %f)\n", xB, yB, dSum);
	memMatB[yB * BWidth + xB] = (float)dSum;
}
