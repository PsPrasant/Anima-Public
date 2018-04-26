#include <animaKummerFunctions.h>
#include <animaBesselFunctions.h>
#include <boost/math/special_functions/gamma.hpp>

namespace anima
{

double
PochHammer(const double &x,
           const unsigned int n)
{
    double resVal = 1.0;
    
    for (unsigned int i = 0;i < n;++i)
        resVal *= (x + i);
    
    return resVal;
}

double
KummerMethod1(const double &x,
              const double &a,
              const double &b,
              const unsigned int maxIter,
              const double tol)
{
    double resVal = 1.0;
    
    bool stopLoop = false;
    unsigned int counter = 0;
    double tVal = 1.0;
    
    while (counter < maxIter && !stopLoop)
    {
        tVal *= (a + counter) * x / (b + counter) / (counter + 1.0);
        resVal += tVal;
        ++counter;
    }
    
    return resVal;
}

double
KummerMethod2(const double &x,
              const double &a,
              const double &b,
              const unsigned int maxIter,
              const double tol)
{
    double resVal = 1.0;
    
    bool stopLoop = false;
    unsigned int counter = 0;
    double factorial_counter = 1;
    
    while (counter < maxIter && !stopLoop)
    {
        ++counter;
        factorial_counter *= counter;
        
        double tmpVal = resVal;
        
        double poch_a, poch_b;
        
        if (x > 0)
        {
            poch_a = PochHammer(1.0 - a,counter);
            poch_b = PochHammer(b - a,counter);
        }
        else
        {
            poch_a = PochHammer(a,counter);
            poch_b = PochHammer(1.0 + a - b,counter);
        }
        
        resVal += poch_a * poch_b * std::pow(std::abs(x),-1.0 * counter) / factorial_counter;
        
        if (std::abs(resVal - tmpVal) < tol)
            stopLoop = true;
    }
    
    if (x > 0)
        resVal *= std::exp(x) * std::pow(x,(double)(a-b)) / boost::math::tgamma(a);
    else
        resVal *= std::pow(-x,-1.0 * a) / boost::math::tgamma(b-a);
    
    resVal *= boost::math::tgamma(b);
    
    return resVal;
}

double
KummerFunction(const double &x,
               const double &a,
               const double &b,
               const unsigned int maxIter,
               const double tol)
{
  if (a == -0.5 && b == 1.0)
    return (x < -600.0) ? std::sqrt(std::abs(x) / M_PI) * 2.0 : (1.0 - x) * anima::scaled_bessel_i(0, -x / 2.0) - x * anima::scaled_bessel_i(1, -x / 2.0);

    if (std::abs(x) < 50.0)
        return KummerMethod1(x,a,b,maxIter,tol);
    else
        return KummerMethod2(x,a,b,maxIter,tol);
}
    
} // end of namespace anima
