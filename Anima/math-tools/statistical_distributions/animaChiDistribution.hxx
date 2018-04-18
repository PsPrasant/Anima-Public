#pragma once

#include "animaChiDistribution.h"
#include <animaKummerFunctions.h>
#include <animaBesselFunctions.h>

#include <itkExceptionObject.h>

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

double LowerBound(const unsigned int N)
{
    double insideValue = 2.0 * N / anima::XiFunction(0, N) - 1.0;
    return std::sqrt(std::max(insideValue, 0.0));
}

double XiFunction(const double eta, const double sigma, const unsigned int N)
{
    double thetaSq = eta * eta / sigma / sigma;
    double betaN = std::sqrt(M_PI / 2.0) * anima::double_factorial(2 * N - 1) / (std::pow(2, N - 1) * anima::factorial(N - 1));
    double tmpVal = betaN * anima::KummerFunction(-thetaSq / 2.0, -0.5, N);
    return (eta > 600.0 * sigma) ? 1.0 : 2.0 * N + thetaSq - tmpVal * tmpVal;
}

double XiFunction(const double theta, const unsigned int N)
{
    double thetaSq = theta * theta;
    double betaN = std::sqrt(M_PI / 2.0) * anima::double_factorial(2 * N - 1) / (std::pow(2, N - 1) * anima::factorial(N - 1));
    double tmpVal = betaN * anima::KummerFunction(-thetaSq / 2.0, -0.5, N);
    
    return (theta > 600.0) ? 1.0 : 2.0 * N + thetaSq - tmpVal * tmpVal;
}

double GFunction(const double eta, const double m, const double sigma, const unsigned int N)
{
    double insideValue = m * m + (anima::XiFunction(eta, sigma, N) - 2 * N) * sigma * sigma;
    return std::sqrt(std::max(insideValue, 0.0));
}

double GFunction(const double theta, const double r, const unsigned int N)
{
    double insideValue = anima::XiFunction(theta, N) * (1.0 + r * r) - 2 * N;
    return std::sqrt(std::max(insideValue, 0.0));
}

double FixedPointFinder(const double m, const double sigma, const unsigned int N, const unsigned int maximumNumberOfIterations, const double epsilon)
{
    unsigned int counter = maximumNumberOfIterations;
    double betaN = std::sqrt(M_PI / 2.0) * anima::double_factorial(2 * N - 1) / (std::pow(2, N - 1) * anima::factorial(N - 1));
    double delta = betaN * sigma - m;
    
    if (delta == 0)
        return 0;
    
    double mCorrected = (delta > 0) ? betaN * sigma + delta : m;
    
    double t0 = mCorrected;
    double t1 = anima::GFunction(t0, mCorrected, sigma, N);
    
    while (std::abs(t0 - t1) > epsilon)
    {
        t0 = t1;
        t1 = anima::GFunction(t0, mCorrected, sigma, N);
        --counter;
        
        if (counter == 0)
            break;
    }
    
    return (delta > 0) ? -t1 : t1;
}

double FixedPointFinder(const double r, const unsigned int N, const unsigned int maximumNumberOfIterations, const double epsilon)
{
    if (r <= anima::LowerBound(N))
        return 0.0;
    
    unsigned int counter = maximumNumberOfIterations;
    
    double t0 = r - anima::LowerBound(N);
    double t1 = anima::GFunction(t0, r, N);
    
    while (std::abs(t0 - t1) > epsilon)
    {
        t0 = t1;
        t1 = anima::GFunction(t0, r, N);
        --counter;
        
        if (counter == 0)
            break;
    }
    
    return t1;
}

void GetRiceParameters(const std::vector<double> &samples, const std::vector<double> &weights, double &location, double &scale)
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
    
    if (scale == 0)
    {
        double sigmaValue = 0;
        
        for (unsigned int i = 0;i < sampleSize;++i)
            sigmaValue += weights[i] * (samples[i] - meanValue) * (samples[i] - meanValue);
        
        sigmaValue /= (sumWeights * sumWeights - sumSqWeights);
        sigmaValue *= sumWeights;
        sigmaValue = std::sqrt(sigmaValue);
        
        double rValue = meanValue / sigmaValue;
        double thetaValue = FixedPointFinder(rValue, 1);
        
        scale = sigmaValue / std::sqrt(XiFunction(thetaValue, 1));
        location = thetaValue * scale;
        return;
    }
    
    location = FixedPointFinder(meanValue, scale, 1);
}

double GetRiceCDF(const double x, const double location, const double scale, const unsigned int gridSize)
{
    double a = location / scale;
    double b = x / scale;
    
    double resVal = 0;
    
    for (unsigned int i = 0;i < gridSize;++i)
    {
        double t = ((double)i / (double)gridSize);
        double integrand = t * std::exp(-(b * t - a) * (b * t - a) / 2.0) * anima::scaled_bessel_i(0, a * b * t);
        resVal += integrand;
    }
    
    resVal *= ((b * b) / gridSize);
    
    return resVal;
}

} // end of namespace anima
