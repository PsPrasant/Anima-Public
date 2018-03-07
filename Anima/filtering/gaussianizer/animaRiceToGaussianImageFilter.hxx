#pragma once
#include <animaRiceToGaussianImageFilter.h>
#include <animaBesselFunctions.h>
#include <animaVectorOperations.h>

#include <itkConstNeighborhoodIterator.h>
#include <itkImageRegionIterator.h>
#include <itkProgressReporter.h>

namespace anima
{

template <unsigned int ImageDimension>
const double
RiceToGaussianImageFilter<ImageDimension>
::m_LowerBound = 2.0 - itk::Math::pi / 2.0;

template <unsigned int ImageDimension>
double
RiceToGaussianImageFilter<ImageDimension>
::VarianceCorrectionFactor(const double theta)
{
	double thetaSq = theta * theta;
	double tmpVal = (2.0 + thetaSq) * anima::scaled_bessel_i(0, thetaSq / 4.0) + thetaSq * anima::scaled_bessel_i(1, thetaSq / 4.0);

    return 2.0 + thetaSq - itk::Math::pi / 8.0 * tmpVal * tmpVal;
}

template <unsigned int ImageDimension>
double
RiceToGaussianImageFilter<ImageDimension>
::FixedPointFormula(const double theta, const double r)
{
	return std::sqrt(std::max(this->VarianceCorrectionFactor(theta) * (1.0 + r * r) - 2.0, 0.0));
}

template <unsigned int ImageDimension>
void
RiceToGaussianImageFilter<ImageDimension>
::GetRiceParameters(const std::vector<double> &samples, const std::vector<double> &weights, double &location, double &scale)
{
	unsigned int sampleSize = samples.size();

	double meanValue = 0;
	double sumWeights = 0, sumSqWeights = 0;
	
	for (unsigned int i = 0;i < sampleSize;++i)
	{
		double weightValue = weights[i];
		meanValue += weightValue * samples[i];
		sumWeights += weightValue;
		sumSqWeights += weightValue * weightValue;
	}

	meanValue /= sumWeights;

	double sigmaValue = 0;

	for (unsigned int i = 0;i < sampleSize;++i)
		sigmaValue += weights[i] * (samples[i] - meanValue) * (samples[i] - meanValue);

	sigmaValue /= (sumWeights * sumWeights - sumSqWeights);
	sigmaValue *= sumWeights;
	sigmaValue = std::sqrt(sigmaValue);

	double rValue = meanValue / sigmaValue;
	double thetaValue = 0.0;

	if (rValue > m_LowerBound)
	{
		thetaValue = rValue - m_LowerBound;
		double newThetaValue = 0.0;
		unsigned int numberOfIterations = 0;

		while (std::abs(newThetaValue - thetaValue) > m_Epsilon && numberOfIterations < m_MaximumNumberOfIterations)
		{
			newThetaValue = thetaValue;
			thetaValue = this->FixedPointFormula(thetaValue, rValue);
			++numberOfIterations;
		}
	}

	scale = sigmaValue / std::sqrt(this->VarianceCorrectionFactor(thetaValue));
	location = std::sqrt(std::max(meanValue * meanValue + (this->VarianceCorrectionFactor(thetaValue) - 2.0) * scale * scale, 0.0));
}

template <unsigned int ImageDimension>
void
RiceToGaussianImageFilter<ImageDimension>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       itk::ThreadIdType threadId)
{
	typename InputImageType::SizeType radius;
    for (unsigned int i = 0;i < ImageDimension;++i)
    	radius[i] = m_Radius;

    typedef itk::ConstNeighborhoodIterator<InputImageType> InputIteratorType;
    typedef itk::ConstNeighborhoodIterator<MaskImageType> MaskIteratorType;
    typedef itk::ImageRegionIterator<OutputImageType> OutputIteratorType;

    InputIteratorType inputItr(radius, this->GetInput(), outputRegionForThread);
    unsigned int neighborhoodSize = inputItr.Size();

    MaskIteratorType maskItr;
    if (m_SegmentationMask)
    	maskItr = MaskIteratorType(radius, m_SegmentationMask, outputRegionForThread);
    
    OutputIteratorType locationItr(this->GetOutput(0), outputRegionForThread);
    OutputIteratorType scaleItr(this->GetOutput(1), outputRegionForThread);
    OutputIteratorType signalItr(this->GetOutput(2), outputRegionForThread);

    std::vector<double> samples, weights;
    bool isInBounds;
    typename InputImageType::IndexType currentIndex, neighborIndex;
    typename InputImageType::PointType currentPoint, neighborPoint;

    // Support for progress methods/callbacks
    itk::ProgressReporter progress(this, threadId, outputRegionForThread.GetNumberOfPixels());

    while (!maskItr.IsAtEnd())
    {
    	if (m_SegmentationMask)
    	{
    		if (maskItr.GetCenterPixel() == 0)
	    	{
	    		locationItr.Set(0);
		        scaleItr.Set(0);
		        signalItr.Set(0);

		        ++inputItr;
		        ++maskItr;
		        ++locationItr;
		        ++scaleItr;
		        ++signalItr;

		        progress.CompletedPixel();
		        continue;
	    	}
    	}

    	double inputSignal = inputItr.GetCenterPixel();

    	if (inputSignal == 0)
    	{
        	locationItr.Set(static_cast<OutputPixelType>(inputSignal));
	        scaleItr.Set(static_cast<OutputPixelType>(0));
	        signalItr.Set(static_cast<OutputPixelType>(inputSignal));

	        ++inputItr;
	        if (m_SegmentationMask)
	        	++maskItr;
	        ++locationItr;
	        ++scaleItr;
	        ++signalItr;

	        progress.CompletedPixel();
	        continue;
    	}

    	samples.clear();
    	weights.clear();
    	currentIndex = inputItr.GetIndex();
    	this->GetInput()->TransformIndexToPhysicalPoint(currentIndex,currentPoint);

		for (unsigned int i = 0; i < neighborhoodSize; ++i)
		{
      		double tmpVal = static_cast<double>(inputItr.GetPixel(i, isInBounds));

		     if (isInBounds && !std::isnan(tmpVal) && std::isfinite(tmpVal))
		     {
		     	if (m_SegmentationMask)
		     		if (maskItr.GetPixel(i) != maskItr.GetCenterPixel())
		     			continue;	

		     	samples.push_back(tmpVal);

		     	neighborIndex = inputItr.GetIndex(i);
    			this->GetInput()->TransformIndexToPhysicalPoint(neighborIndex,neighborPoint);

    			double distanceVal = anima::ComputeNorm(currentPoint - neighborPoint);
    			double weight = std::exp(-distanceVal * distanceVal / (2.0 * m_Sigma * m_Sigma));

    			weights.push_back(weight);
		     }	
		}

        double location, scale, gaussSignal;

        if (samples.size() == 1)
        {
        	location = inputSignal;
        	scale = 0;
        	gaussSignal = location;
        }
        else
        {
        	this->GetRiceParameters(samples, weights, location, scale);
        	double unifSignal = anima::EvaluateRiceCDF(inputSignal, location, scale);
	    	gaussSignal = location + scale * boost::math::quantile(m_NormalDistribution, unifSignal);
        }

        locationItr.Set(static_cast<OutputPixelType>(location));
        scaleItr.Set(static_cast<OutputPixelType>(scale));
        signalItr.Set(static_cast<OutputPixelType>(gaussSignal));

        ++inputItr;
        if (m_SegmentationMask)
        	++maskItr;
        ++locationItr;
        ++scaleItr;
        ++signalItr;

        progress.CompletedPixel();
    }
}

} // end of namespace anima
