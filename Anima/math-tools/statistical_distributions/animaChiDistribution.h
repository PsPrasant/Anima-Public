#pragma once

#include <vector>

namespace anima
{

//! Implementation of Koay and Basser, Analytically exact correction scheme for signal extraction from noisy magnitude MR signals, Journal of Magnetic Resonance (2006).
double LowerBound(const unsigned int N);
double XiFunction(const double theta, const unsigned int N);
double GFunction(const double theta, const double r, const unsigned int N);
double FixedPointFinder(const double r, const unsigned int N, const unsigned int maximumNumberOfIterations = 500, const double epsilon = 1.0e-9);

//! Implementation of Koay, Ozarslan and Basser, A signal transformational framework for breaking the noise floor and its applications in MRI, Journal of Magnetic Resonance (2009).
double XiFunction(const double eta, const double sigma, const unsigned int N);
double GFunction(const double eta, const double m, const double sigma, const unsigned int N);
double FixedPointFinder(const double m, const double sigma, const unsigned int N, const unsigned int maximumNumberOfIterations = 500, const double epsilon = 1.0e-9);

//! Implementation of both Koay technique for Rice parameter estimation
void GetRiceParameters(const std::vector<double> &samples, const std::vector<double> &weights, double &location, double &scale);

//! In-house function for evaluating the cumulative distribution function of a Rice distribution based on rectangle integral approximation of the Marcum Q function.
double GetRiceCDF(const double x, const double location, const double scale, const unsigned int gridSize = 2000);
    
} // end of namespace

#include "animaChiDistribution.hxx"
