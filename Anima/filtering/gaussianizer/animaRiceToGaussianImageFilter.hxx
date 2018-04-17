#pragma once
#include <animaRiceToGaussianImageFilter.h>
#include <animaChiDistribution.h>
#include <animaVectorOperations.h>

#include <itkConstNeighborhoodIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <itkProgressReporter.h>
#include <itkGaussianOperator.h>

namespace anima
{

template <unsigned int ImageDimension>
void
RiceToGaussianImageFilter<ImageDimension>
::BeforeThreadedGenerateData(void)
{
    // Compute spatial weights of neighbors beforehand
    typename InputImageType::SpacingType spacing = this->GetInput()->GetSpacing();
    typedef itk::GaussianOperator<double,ImageDimension> GaussianOperatorType;
    std::vector<GaussianOperatorType> gaussianKernels(ImageDimension);
    
    for (unsigned int i = 0;i < ImageDimension;++i)
    {
        unsigned int reverse_i = ImageDimension - i - 1;
        double stddev = m_Sigma / spacing[i];
        gaussianKernels[reverse_i].SetDirection(i);
        gaussianKernels[reverse_i].SetVariance(stddev * stddev);
        gaussianKernels[reverse_i].SetMaximumError(1e-3);
        gaussianKernels[reverse_i].CreateDirectional();
        gaussianKernels[reverse_i].ScaleCoefficients(1.0e4);
        m_Radius[i] = gaussianKernels[reverse_i].GetRadius(i);
    }
    
    m_NeighborWeights.clear();
    for (unsigned int i = 0;i < gaussianKernels[0].Size();++i)
    {
        for (unsigned int j = 0;j < gaussianKernels[1].Size();++j)
        {
            for (unsigned int k = 0;k < gaussianKernels[2].Size();++k)
            {
                double weight = gaussianKernels[0][i] * gaussianKernels[1][j] * gaussianKernels[2][k];
                m_NeighborWeights.push_back(weight);
            }
        }
    }
    
    // Initialize thread containers for global scale estimation
    unsigned int numThreads = this->GetNumberOfThreads();
    m_ThreadScaleSamples.resize(numThreads);
    this->Superclass::BeforeThreadedGenerateData();
}

template <unsigned int ImageDimension>
void
RiceToGaussianImageFilter<ImageDimension>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       itk::ThreadIdType threadId)
{
    typedef itk::ConstNeighborhoodIterator<InputImageType> InputIteratorType;
    typedef itk::ConstNeighborhoodIterator<MaskImageType> MaskIteratorType;
    typedef itk::ImageRegionConstIterator<InputImageType> InputSimpleIteratorType;
    typedef itk::ImageRegionIterator<OutputImageType> OutputIteratorType;
    
    InputIteratorType inputItr(m_Radius, this->GetInput(), outputRegionForThread);
    unsigned int neighborhoodSize = inputItr.Size();
    
    MaskIteratorType maskItr;
    if (m_SegmentationMask)
        maskItr = MaskIteratorType(m_Radius, m_SegmentationMask, outputRegionForThread);
    
    InputSimpleIteratorType meanItr, varItr;
    if (m_MeanImage)
        meanItr = InputSimpleIteratorType(m_MeanImage, outputRegionForThread);
    if (m_VarianceImage)
        varItr = InputSimpleIteratorType(m_VarianceImage, outputRegionForThread);
    
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
            // Discard voxels outside of the brain
            if (maskItr.GetCenterPixel() == 0)
            {
                locationItr.Set(0);
                scaleItr.Set(0);
                signalItr.Set(0);
                
                ++inputItr;
                ++maskItr;
                if (m_MeanImage)
                    ++meanItr;
                if (m_VarianceImage)
                    ++varItr;
                ++locationItr;
                ++scaleItr;
                ++signalItr;
                
                progress.CompletedPixel();
                continue;
            }
        }
        
        // Get signal at current central voxel
        double inputSignal = inputItr.GetCenterPixel();
        
        // Rice-corrupted signals should all be positive
        if (inputSignal <= 0)
        {
            locationItr.Set(static_cast<OutputPixelType>(0));
            scaleItr.Set(static_cast<OutputPixelType>(0));
            signalItr.Set(static_cast<OutputPixelType>(0));
            
            ++inputItr;
            if (m_SegmentationMask)
                ++maskItr;
            if (m_MeanImage)
                ++meanItr;
            if (m_VarianceImage)
                ++varItr;
            ++locationItr;
            ++scaleItr;
            ++signalItr;
            
            progress.CompletedPixel();
            continue;
        }
        
        // Estimation of location and scale
        double location = 0;
        double scale = m_Scale;
        
        if (m_MeanImage && m_VarianceImage)
        {
            // If mean and variance images are available, use them instead of neighborhood
            double sigmaValue = std::sqrt(varItr.Get());
            double rValue = meanItr.Get() / sigmaValue;
            double thetaValue = anima::FixedPointFinder(rValue, 1);
            
            scale = sigmaValue / std::sqrt(anima::XiFunction(thetaValue, 1));
            location = thetaValue * scale;
        }
        else
        {
            // Use neighbors to create samples
            samples.clear();
            weights.clear();
            
            for (unsigned int i = 0; i < neighborhoodSize; ++i)
            {
                double tmpVal = static_cast<double>(inputItr.GetPixel(i, isInBounds));
                
                if (isInBounds && !std::isnan(tmpVal) && std::isfinite(tmpVal))
                {
                    if (m_SegmentationMask)
                        if (maskItr.GetPixel(i) != maskItr.GetCenterPixel())
                            continue;
                    
                    double weight = m_NeighborWeights[i];
                    
                    if (weight < m_Epsilon)
                        continue;
                    
                    samples.push_back(tmpVal);
                    weights.push_back(weight);
                }
            }
            
            if (samples.size() == 1)
                location = inputSignal;
            else
                anima::GetRiceParameters(samples, weights, location, scale);
        }
        
        // Transform Rice signal in Gaussian signal
        double outputSignal = inputSignal;
        double snrValue = location / scale;
        
        if (!std::isfinite(snrValue) || std::isnan(snrValue))
            itkExceptionMacro("Estimated SNR is invalid");
        
        if (snrValue <= 0.1)
        {
            double unifSignal = boost::math::cdf(m_RayleighDistribution, inputSignal / scale);
            
            if (unifSignal >= 1.0 - m_Epsilon || unifSignal <= m_Epsilon)
            {
                outputSignal = location;
                std::cout << "Outlier: " << inputItr.GetIndex() << std::endl;
            }
            else
                outputSignal = scale * boost::math::quantile(m_NormalDistribution, unifSignal);
        }
        else if (snrValue <= 600) // if SNR if > 600 keep signal as is, else...
        {
            double unifSignal = anima::EvaluateRiceCDF(inputSignal, location, scale);
            
            if (unifSignal >= 1.0 - m_Epsilon || unifSignal <= m_Epsilon)
            {
                outputSignal = location;
                std::cout << "Outlier: " << inputItr.GetIndex() << std::endl;
            }
            else
                outputSignal = location + scale * boost::math::quantile(m_NormalDistribution, unifSignal);
        }
        
        m_ThreadScaleSamples[threadId].push_back(scale);
        
        locationItr.Set(static_cast<OutputPixelType>(location));
        scaleItr.Set(static_cast<OutputPixelType>(scale));
        signalItr.Set(static_cast<OutputPixelType>(outputSignal));
        
        ++inputItr;
        if (m_SegmentationMask)
            ++maskItr;
        if (m_MeanImage)
            ++meanItr;
        if (m_VarianceImage)
            ++varItr;
        ++locationItr;
        ++scaleItr;
        ++signalItr;
        
        progress.CompletedPixel();
    }
}

template <unsigned int ImageDimension>
void
RiceToGaussianImageFilter<ImageDimension>
::AfterThreadedGenerateData(void)
{
    if (m_Scale == 0)
    {
        std::vector<double> scaleSamples;
        for (unsigned int i = 0;i < this->GetNumberOfThreads();++i)
            scaleSamples.insert(scaleSamples.end(), m_ThreadScaleSamples[i].begin(), m_ThreadScaleSamples[i].end());
        
        m_Scale = anima::GetMedian(scaleSamples);
    }
    
    m_ThreadScaleSamples.clear();
    m_NeighborWeights.clear();
    
    this->Superclass::AfterThreadedGenerateData();
}
    
} // end of namespace anima
