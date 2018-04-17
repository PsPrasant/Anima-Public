#pragma once

#include <vector>

namespace anima
{
    
double LowerBound(const unsigned int N);
double XiFunction(const double eta, const double sigma, const unsigned int N);
double XiFunction(const double theta, const unsigned int N);
double GFunction(const double eta, const double m, const double sigma, const unsigned int N);
double GFunction(const double theta, const double r, const unsigned int N);
double FixedPointFinder(const double m, const double sigma, const unsigned int N, const unsigned int maximumNumberOfIterations = 500, const double epsilon = 1.0e-9);
double FixedPointFinder(const double r, const unsigned int N, const unsigned int maximumNumberOfIterations = 500, const double epsilon = 1.0e-9);
void GetRiceParameters(const std::vector<double> &samples, const std::vector<double> &weights, double &location, double &scale);
// Compute cumulative distribution function of Rice distribution (look at https://github.com/cscooper/ClusterLib/blob/master/src/MarcumQ.cc for generalized Marcum Q function computation)
double EvaluateRiceCDF(const double x, const double location, const double scale);
    
} // end of namespace

#include "animaChiDistribution.hxx"
