#define TYPE_FLOAT float

TYPE_FLOAT SmoothStep(TYPE_FLOAT xMin, TYPE_FLOAT xMax, TYPE_FLOAT x);

TYPE_FLOAT DetailRemap(const TYPE_FLOAT alpha, TYPE_FLOAT delta, TYPE_FLOAT sigmaR)
{
	TYPE_FLOAT fraction = delta / sigmaR; // 0.0 / 1.0
	TYPE_FLOAT polynomial = pow(fraction, alpha); // pow(0.0, 1.0)
	if (alpha < (TYPE_FLOAT)1.0) // always false because (1.0 < 1.0)
	{
		const TYPE_FLOAT kNoiseLevel = (TYPE_FLOAT)0.01;
		TYPE_FLOAT blend = SmoothStep(kNoiseLevel, (TYPE_FLOAT)2.0 * kNoiseLevel, fraction * sigmaR);
		polynomial = blend * polynomial + ((TYPE_FLOAT)1.0 - blend) * fraction;
	}
	return polynomial; // 0.0
}

TYPE_FLOAT EdgeRemap(const TYPE_FLOAT beta, TYPE_FLOAT delta)
{
	return beta * delta;
}

TYPE_FLOAT SmoothStep(TYPE_FLOAT xMin, TYPE_FLOAT xMax, TYPE_FLOAT x)
{
	TYPE_FLOAT y = (x - xMin) / (xMax - xMin);
	y = fmax((TYPE_FLOAT)0.0, fmin((TYPE_FLOAT)1.0, y));
	return pow(y, (TYPE_FLOAT)2.0) * pow(y - (TYPE_FLOAT)2.0, (TYPE_FLOAT)2.0);
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

	// comment - (2022-05-26, Ethan) no need for this sample program
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

	__global const float* pfG0Local = (__global const float*)(memMatG0Data + nROITop * G0Width + nROILeft);

	memCoeffGRData += matCoeffGRWidth * yB;
	memCoeffGUpRData += matCoeffGUpRWidth * yB;

	memCoeffGCData += matCoeffGCWidth * xB;
	memCoeffGUpCData += matCoeffGUpCWidth * xB;

	const TYPE_FLOAT reference = (TYPE_FLOAT)memMatG[yB * BWidth + xB]; // should be 0.0

	TYPE_FLOAT dSum = (TYPE_FLOAT)0.0;
	for (int nY = 0; nY < nROIHeight; nY++)
	{
		for (int nX = 0; nX < nROIWidth; nX++)
		{
			TYPE_FLOAT value = pfG0Local[nY * G0Width + nX]; // should be 0.0
			TYPE_FLOAT delta = fabs(value - reference); // should be 0.0
			TYPE_FLOAT sign = value < reference ? (TYPE_FLOAT)-1.0 : (TYPE_FLOAT)1.0; // should be 1.0
			TYPE_FLOAT output = (TYPE_FLOAT)0.0;
			if (delta < sigmaR) // should be true because (0.0 < 1.0) 
				// should be 0.0 = 0.0 + 1.0 * 1.0 * DetailRemap(1.0, 0.0, 1.0);
				// should be 0.0 = 0.0 + 1.0 * 1.0 * 0.0
				output = reference + sign * sigmaR * DetailRemap(alpha, delta, sigmaR);
			else
				// next line should not be called
				output = reference + sign * (EdgeRemap(beta, delta - sigmaR) + sigmaR);
			dSum += (memCoeffGCData[nX] * memCoeffGRData[nY] - memCoeffGUpCData[nX] * memCoeffGUpRData[nY]) * output;
			// should be dSum += (0.0 * 0.0 - 0.0 * 0.0) * 0.0
			// should be dSum += 0.0
		}
	}

	printf("(%d, %d, %f)\n", xB, yB, dSum);
	memMatB[yB * BWidth + xB] = (float)dSum;
}
