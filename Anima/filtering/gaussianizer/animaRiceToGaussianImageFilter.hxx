#pragma once
#include <animaRiceToGaussianImageFilter.h>
#include <animaBesselFunctions.h>
#include <animaKummerFunctions.h>
#include <animaVectorOperations.h>

#include <itkConstNeighborhoodIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <itkProgressReporter.h>
#include <itkGaussianOperator.h>

#include <boost/math/distributions/rayleigh.hpp>
#include <boost/math/special_functions/bessel.hpp>

namespace anima
{
    
inline double factorial(unsigned int x)
{
    return (x <= 1) ? 1 : x * factorial(x - 1);
}

inline double double_factorial(unsigned int x)
{
    return (x <= 1) ? 1 : x * double_factorial(x - 2);
}

inline double MarcumQ( double a, double b, int M ) {
    
    
    // Special cases.
    if ( b == 0 )
        return 1;
    
    if ( a == 0 ) {
        double Q=0;
        for ( int k = 0; k <= M-1; k++ )
            Q += pow(b,2*k)/(pow(2.0,k) * factorial(k));
        return Q * exp(-pow(b,2)/2);
    }
    
    // The basic iteration.  If a<b compute Q_M, otherwise compute 1-Q_M.
    double qSign;
    double constant;
    int k = M;
    double z = a*b;
    double t = 1, d, S = 0, x;
    if ( a < b ) {
        
        qSign = +1;
        constant = 0;
        x = a / b;
        d = x;
        k = 0;
        S = anima::scaled_bessel_i(0, z);
        
        for ( k = 1; k <= M-1; k++ ) {
            t = ( d + 1/d ) * anima::scaled_bessel_i(k, z);
            S += t;
            d *= x;
        }
        
        k = M;
        
    } else {
        
        qSign = -1;
        constant = 1;
        x = b / a;
        d = pow( x, M );
        
    }
    
    do {
        t = d * anima::scaled_bessel_i(k, z);
        S += t;
        d *= x;
        k++;
    } while ( fabs(t/S) > 1.0e-3 );// std::numeric_limits<double>::epsilon() );
    
    return constant + qSign * exp( -pow( a-b, 2 ) / 2 ) * S;
    
}

inline double GetMode(const std::vector<double> &data)
{
    double scaling_factor = 1e9;
    unsigned int dimension = data.size();
    std::vector <unsigned int> dataInt(dimension,0);
    
    for (unsigned int i = 0;i < dimension;++i)
        dataInt[i] = (data[i] * scaling_factor);
    
    unsigned int mode = dataInt[0];
    unsigned int modeCounter = 0;
    unsigned int currentEl = dataInt[0];
    unsigned int currCounter = 0;
    
    for (unsigned int i = 0;i < dimension;++i)
    {
        unsigned int e = dataInt[i];
        
        if (e == currentEl)
            ++currCounter;
        else
        {
            if (currCounter > modeCounter)
            {
                mode = currentEl;
                modeCounter = currCounter;
            }
            
            currentEl = e;
            currCounter = 1;
        }
    }
    
    if (currCounter > modeCounter)
    {
        mode = currentEl;
        modeCounter = currCounter;
    }
    
    return (double)mode / scaling_factor;
}

inline double GetMedian(const std::vector<double> &data)
{
    unsigned int dimension = data.size();
    std::vector <double> array(dimension,0);
    
    for (unsigned int i = 0;i < dimension;++i)
        array[i] = data[i];
    
    std::nth_element(array.begin(), array.begin() + dimension / 2, array.end());
    
    double median = array[dimension / 2];
    
    if (dimension % 2 == 0)
    {
        median += *std::max_element(array.begin(), array.begin() + dimension / 2);
        median /= 2.0;
    }
    
    return median;
}

template <unsigned int ImageDimension>
double
RiceToGaussianImageFilter<ImageDimension>
::LowerBound(const unsigned int N)
{
    double insideValue = 2.0 * N / this->XiFunction(0, N) - 1.0;
    return std::sqrt(std::max(insideValue, 0.0));
}

template <unsigned int ImageDimension>
double
RiceToGaussianImageFilter<ImageDimension>
::XiFunction(const double eta, const double sigma, const unsigned int N)
{
    double thetaSq = eta * eta / sigma / sigma;
    double betaN = std::sqrt(itk::Math::pi / 2.0) * double_factorial(2 * N - 1) / (std::pow(2, N - 1) * factorial(N - 1));
    double tmpVal = betaN * anima::KummerFunction(-thetaSq / 2.0, -0.5, N);
    return (eta > 600 * sigma) ? 1.0 : 2.0 * N + thetaSq - tmpVal * tmpVal;
}

template <unsigned int ImageDimension>
double
RiceToGaussianImageFilter<ImageDimension>
::XiFunction(const double theta, const unsigned int N)
{
    double thetaSq = theta * theta;
    double betaN = std::sqrt(itk::Math::pi / 2.0) * double_factorial(2 * N - 1) / (std::pow(2, N - 1) * factorial(N - 1));
    double tmpVal = betaN * anima::KummerFunction(-thetaSq / 2.0, -0.5, N);
    
    return (theta > 600.0) ? 1.0 : 2.0 * N + thetaSq - tmpVal * tmpVal;
}

template <unsigned int ImageDimension>
double
RiceToGaussianImageFilter<ImageDimension>
::GFunction(const double eta, const double m, const double sigma, const unsigned int N)
{
    double insideValue = m * m + (this->XiFunction(eta, sigma, N) - 2 * N) * sigma * sigma;
    return std::sqrt(std::max(insideValue, 0.0));
}

template <unsigned int ImageDimension>
double
RiceToGaussianImageFilter<ImageDimension>
::GFunction(const double theta, const unsigned int N, const double r)
{
    double insideValue = this->XiFunction(theta, N) * (1.0 + r * r) - 2 * N;
    return std::sqrt(std::max(insideValue, 0.0));
}

template <unsigned int ImageDimension>
double
RiceToGaussianImageFilter<ImageDimension>
::KFunction(const double eta, const double m, const double sigma, const unsigned int N)
{
    double numeratorValue = this->GFunction(eta, m, sigma, N) * (this->GFunction(eta, m, sigma, N) - eta);
    double betaN = std::sqrt(itk::Math::pi / 2.0) * double_factorial(2 * N - 1) / (std::pow(2, N - 1) * factorial(N - 1));
    double thetaSq = eta * eta / (2.0 * sigma * sigma);
    double denominatorValue = eta * (1 - betaN * betaN * anima::KummerFunction(-thetaSq, -0.5, N) * anima::KummerFunction(-thetaSq, 0.5, N + 1) / (2.0 * N)) - this->GFunction(eta, m, sigma, N);
    return eta - numeratorValue / denominatorValue;
}

template <unsigned int ImageDimension>
double
RiceToGaussianImageFilter<ImageDimension>
::KFunction(const double theta, const unsigned int N, const double r)
{
    double numeratorValue = this->GFunction(theta, N, r) * (this->GFunction(theta, N, r) - theta);
    double betaN = std::sqrt(itk::Math::pi / 2.0) * double_factorial(2 * N - 1) / (std::pow(2, N - 1) * factorial(N - 1));
    double thetaSq = theta * theta;
    double denominatorValue = theta * (1.0 + r * r) * (1 - betaN * betaN * anima::KummerFunction(-thetaSq / 2.0, -0.5, N) * anima::KummerFunction(-thetaSq / 2.0, 0.5, N + 1) / (2.0 * N)) - this->GFunction(theta, N, r);
    return theta - numeratorValue / denominatorValue;
}

template <unsigned int ImageDimension>
double
RiceToGaussianImageFilter<ImageDimension>
::FixedPointFinder(const double m, const double sigma, const unsigned int N)
{
    unsigned int counter = m_MaximumNumberOfIterations;
    double betaN = std::sqrt(itk::Math::pi / 2.0) * double_factorial(2 * N - 1) / (std::pow(2, N - 1) * factorial(N - 1));
    double delta = betaN * sigma - m;
    
    if (delta == 0)
        return 0;
    
    double mCorrected = (delta > 0) ? betaN * sigma + delta : m;
    
    double t0 = mCorrected;
    // double t1 = this->KFunction(t0, mCorrected, sigma, N);
    double t1 = this->GFunction(t0, mCorrected, sigma, N);
    
    while (std::abs(t0 - t1) > m_Epsilon)
    {
        t0 = t1;
        // t1 = this->KFunction(t0, mCorrected, sigma, N);
        t1 = this->GFunction(t0, mCorrected, sigma, N);
        --counter;
        
        if (counter == 0)
            break;
    }
    
    return (delta > 0) ? -t1 : t1;
}

template <unsigned int ImageDimension>
double
RiceToGaussianImageFilter<ImageDimension>
::FixedPointFinder(const double r, const unsigned int N)
{
    if (r <= this->LowerBound(N))
        return 0.0;
    
    unsigned int counter = m_MaximumNumberOfIterations;
    
    double t0 = r - this->LowerBound(N);
    // double t1 = this->KFunction(t0, N, r);
    double t1 = this->GFunction(t0, N, r);
    
    while (std::abs(t0 - t1) > m_Epsilon)
    {
        t0 = t1;
        // t1 = this->KFunction(t0, N, r);
        t1 = this->GFunction(t0, N, r);
        --counter;
        
        if (counter == 0)
            break;
    }
    
    return t1;
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
    
    if (m_Scale == 0)
    {
        double sigmaValue = 0;
        
        for (unsigned int i = 0;i < sampleSize;++i)
            sigmaValue += weights[i] * (samples[i] - meanValue) * (samples[i] - meanValue);
        
        sigmaValue /= (sumWeights * sumWeights - sumSqWeights);
        sigmaValue *= sumWeights;
        sigmaValue = std::sqrt(sigmaValue);
        
        double rValue = meanValue / sigmaValue;
        double thetaValue = this->FixedPointFinder(rValue, 1);
        
        scale = sigmaValue / std::sqrt(this->XiFunction(thetaValue, 1));
        location = thetaValue * scale;
        return;
    }
    
    double thetaValue = 0.0;
    location = this->FixedPointFinder(meanValue, m_Scale, 1);
    scale = m_Scale;
    
    if (std::isnan(scale) || std::isnan(location) || !std::isfinite(scale) || !std::isfinite(location))
    {
        std::cout << "Bunch of stuff: " << sampleSize << " " << " " << thetaValue << " " << this->XiFunction(thetaValue, 1) << " " << meanValue << " " << scale << std::endl;
        std::cout << "Samples: ";
        for (unsigned int i = 0;i < sampleSize;++i)
            std::cout << samples[i] << " ";
        std::cout << std::endl;
        std::cout << "Weights: ";
        for (unsigned int i = 0;i < sampleSize;++i)
            std::cout << weights[i] << " ";
        std::cout << std::endl;
    }
}

template <unsigned int ImageDimension>
void
RiceToGaussianImageFilter<ImageDimension>
::BeforeThreadedGenerateData(void)
{
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
        
        double inputSignal = inputItr.GetCenterPixel();
        
        if (inputSignal <= 0)
        {
            locationItr.Set(static_cast<OutputPixelType>(inputSignal));
            scaleItr.Set(static_cast<OutputPixelType>(0));
            signalItr.Set(static_cast<OutputPixelType>(inputSignal));
            
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
        
        double location, scale, gaussSignal;
        
        if (m_MeanImage && m_VarianceImage)
        {
            double sigmaValue = std::sqrt(varItr.Get());
            double rValue = meanItr.Get() / sigmaValue;
            double thetaValue = this->FixedPointFinder(rValue, 1);
            
            scale = sigmaValue / std::sqrt(this->XiFunction(thetaValue, 1));
            location = thetaValue * scale;
            
            if (location / scale > 600)
                gaussSignal = location;
            else if (location / scale < 0.1)
            {
                boost::math::rayleigh_distribution<> rayleighDist(scale);
                double unifSignal = boost::math::cdf(rayleighDist, inputSignal);
                
                if (unifSignal >= 1.0 - m_Epsilon || unifSignal <= m_Epsilon)
                    gaussSignal = location;
                else
                    gaussSignal = scale * boost::math::quantile(m_NormalDistribution, unifSignal);
            }
            else
            {
                double unifSignal = anima::EvaluateRiceCDF(inputSignal, location, scale);
                // double unifSignal = MarcumQ(location / scale, inputSignal / scale, 1);
                
                if (unifSignal >= 1.0 - m_Epsilon || unifSignal <= m_Epsilon)
                    gaussSignal = location;
                else
                    gaussSignal = location + scale * boost::math::quantile(m_NormalDistribution, unifSignal);
            }
        }
        else
        {
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
            {
                location = inputSignal;
                scale = m_Scale;
                gaussSignal = location;
            }
            else
            {
                this->GetRiceParameters(samples, weights, location, scale);
                
                if (location / scale > 600)
                    gaussSignal = location;
                else if (location / scale < 0.1)
                {
                    boost::math::rayleigh_distribution<> rayleighDist(scale);
                    double unifSignal = boost::math::cdf(rayleighDist, inputSignal);
                    
                    if (unifSignal >= 1.0 - m_Epsilon || unifSignal <= m_Epsilon)
                        gaussSignal = location;
                    else
                        gaussSignal = scale * boost::math::quantile(m_NormalDistribution, unifSignal);
                }
                else
                {
                    double unifSignal = anima::EvaluateRiceCDF(inputSignal, location, scale);
                    // double unifSignal = MarcumQ(location / scale, inputSignal / scale, 1);
                    
                    if (unifSignal >= 1.0 - m_Epsilon || unifSignal <= m_Epsilon)
                        gaussSignal = location;
                    else
                        gaussSignal = location + scale * boost::math::quantile(m_NormalDistribution, unifSignal);
                }
            }
        }
        
        m_ThreadScaleSamples[threadId].push_back(scale);
        
        locationItr.Set(static_cast<OutputPixelType>(location));
        scaleItr.Set(static_cast<OutputPixelType>(scale));
        signalItr.Set(static_cast<OutputPixelType>(gaussSignal));
        
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
        
        m_Scale = GetMedian(scaleSamples);
    }
    
    m_ThreadScaleSamples.clear();
    m_NeighborWeights.clear();
    
    this->Superclass::AfterThreadedGenerateData();
}
    
} // end of namespace anima
