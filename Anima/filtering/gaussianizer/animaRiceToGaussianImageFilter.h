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
    typedef typename MaskImageType::SizeType SizeType;
    
    typedef typename InputImageType::Pointer InputPointerType;
    typedef typename OutputImageType::Pointer OutputPointerType;
    
    itkSetMacro(MaximumNumberOfIterations, unsigned int)
    itkSetMacro(Epsilon, double)
    itkSetMacro(Sigma, double)
    itkSetMacro(SegmentationMask, MaskPointerType)
    itkSetMacro(MeanImage, InputPointerType)
    itkSetMacro(VarianceImage, InputPointerType)
    
    itkSetMacro(Scale, double)
    itkGetConstMacro(Scale, double)
    
    OutputPointerType GetLocationImage() {return this->GetOutput(0);}
    OutputPointerType GetScaleImage() {return this->GetOutput(1);}
    OutputPointerType GetGaussianImage() {return this->GetOutput(2);}
    
protected:
    RiceToGaussianImageFilter()
    : Superclass()
    {
        m_MaximumNumberOfIterations = 100;
        m_Epsilon = 1.0e-8;
        m_Sigma = 1.0;
        m_SegmentationMask = NULL;
        m_Scale = 0.0;
        m_ThreadScaleSamples.clear();
        m_NeighborWeights.clear();
        m_Radius.Fill(0);
        m_MeanImage = NULL;
        m_VarianceImage = NULL;
        
        this->SetNumberOfRequiredOutputs(3);
        this->SetNthOutput(0, this->MakeOutput(0));
        this->SetNthOutput(1, this->MakeOutput(1));
        this->SetNthOutput(2, this->MakeOutput(2));
    }
    
    virtual ~RiceToGaussianImageFilter() {}
    
    void BeforeThreadedGenerateData(void) ITK_OVERRIDE;
    void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                              itk::ThreadIdType threadId) ITK_OVERRIDE;
    void AfterThreadedGenerateData(void) ITK_OVERRIDE;
    
private:
    ITK_DISALLOW_COPY_AND_ASSIGN(RiceToGaussianImageFilter);
    
    double LowerBound(const unsigned int N);
    double XiFunction(const double eta, const double sigma, const unsigned int N);
    double XiFunction(const double theta, const unsigned int N);
    double GFunction(const double eta, const double m, const double sigma, const unsigned int N);
    double GFunction(const double theta, const unsigned int N, const double r);
    double KFunction(const double eta, const double m, const double sigma, const unsigned int N);
    double KFunction(const double theta, const unsigned int N, const double r);
    double FixedPointFinder(const double m, const double sigma, const unsigned int N);
    double FixedPointFinder(const double r, const unsigned int N);
    void GetRiceParameters(const std::vector<double> &samples, const std::vector<double> &weights, double &location, double &scale);
    
    unsigned int m_MaximumNumberOfIterations;
    double m_Epsilon, m_Sigma, m_Scale;
    MaskPointerType m_SegmentationMask;
    std::vector<std::vector<double> > m_ThreadScaleSamples;
    SizeType m_Radius;
    std::vector<double> m_NeighborWeights;
    InputPointerType m_MeanImage, m_VarianceImage;
    
    boost::math::normal_distribution<> m_NormalDistribution;
};
    
} // end of namespace anima

#include "animaRiceToGaussianImageFilter.hxx"
