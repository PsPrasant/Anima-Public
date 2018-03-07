#pragma once

#include <itkImageToImageFilter.h>

#include <boost/math/distributions/normal.hpp>

namespace anima
{
template <unsigned int ImageDimension>
class RiceToGaussianImageFilter :
    public itk::ImageToImageFilter<itk::Image<float,ImageDimension>,itk::Image<float,ImageDimension> >
{
public:
    /** Standard class typedefs. */
	typedef RiceToGaussianImageFilter Self;
    
    typedef itk::Image<float,ImageDimension> InputImageType;
    typedef typename InputImageType::PixelType InputPixelType;
    
    typedef itk::Image<float,ImageDimension> OutputImageType;
    typedef typename OutputImageType::PixelType OutputPixelType;
    
    typedef itk::ImageToImageFilter<InputImageType, OutputImageType> Superclass;
    
	typedef itk::SmartPointer<Self> Pointer;
	typedef itk::SmartPointer<const Self>  ConstPointer;
	
	/** Method for creation through the object factory. */
	itkNewMacro(Self)
	
	/** Run-time type information (and related methods) */
	itkTypeMacro(RiceToGaussianImageFilter, ImageToImageFilter)
	
	/** Superclass typedefs. */
	typedef typename Superclass::InputImageRegionType InputImageRegionType;
	typedef typename Superclass::OutputImageRegionType OutputImageRegionType;

	typedef itk::Image<unsigned char,ImageDimension> MaskImageType;
    typedef typename MaskImageType::Pointer MaskPointerType;

    itkSetMacro(Radius, unsigned int)
    itkSetMacro(MaximumNumberOfIterations, unsigned int)
    itkSetMacro(Epsilon, double)
    itkSetMacro(Sigma, double)
    itkSetMacro(SegmentationMask, MaskPointerType)

    typename OutputImageType::Pointer GetLocationImage() {return this->GetOutput(0);}
    typename OutputImageType::Pointer GetScaleImage() {return this->GetOutput(1);}
    typename OutputImageType::Pointer GetGaussianImage() {return this->GetOutput(2);}

protected:
    RiceToGaussianImageFilter()
    : Superclass()
	{
        m_Radius = 1;
        m_MaximumNumberOfIterations = 100;
        m_Epsilon = 1.0e-8;
        m_Sigma = 1.0;
	    m_SegmentationMask = NULL;

	    this->SetNumberOfRequiredOutputs(3);
	    this->SetNthOutput(0, this->MakeOutput(0));
	    this->SetNthOutput(1, this->MakeOutput(1));
	    this->SetNthOutput(2, this->MakeOutput(2));
	}

    virtual ~RiceToGaussianImageFilter() {}

    void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                              itk::ThreadIdType threadId) ITK_OVERRIDE;

private:
    ITK_DISALLOW_COPY_AND_ASSIGN(RiceToGaussianImageFilter);

    double VarianceCorrectionFactor(const double theta);
    double FixedPointFormula(const double theta, const double r);
    void GetRiceParameters(const std::vector<double> &samples, const std::vector<double> &weights, double &location, double &scale);

    unsigned int m_Radius, m_MaximumNumberOfIterations;
    double m_Epsilon, m_Sigma;
    MaskPointerType m_SegmentationMask;

    boost::math::normal_distribution<> m_NormalDistribution;
    static const double m_LowerBound;
};

} // end of namespace anima

#include "animaRiceToGaussianImageFilter.hxx"
