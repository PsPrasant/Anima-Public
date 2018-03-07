#include <animaRiceToGaussianImageFilter.h>
#include <animaReadWriteFunctions.h>
#include <tclap/CmdLine.h>

#include <itkImage.h>
#include <itkCommand.h>

//Update progression of the process
void eventCallback (itk::Object* caller, const itk::EventObject& event, void* clientData)
{
    itk::ProcessObject * processObject = (itk::ProcessObject*) caller;
    std::cout<<"\033[K\rProgression: "<<(int)(processObject->GetProgress() * 100)<<"%"<<std::flush;
}

int main(int ac, const char** av)
{

    TCLAP::CmdLine cmd("INRIA / IRISA - VisAGeS Team", ' ',ANIMA_VERSION);

    TCLAP::ValueArg<std::string> inputArg("i",
                                          "input",
                                          "Input Rice-corrupted image",
                                          true,
                                          "",
                                          "Input Rice-corrupted image",
                                          cmd);

    TCLAP::ValueArg<std::string> outputArg("o",
                                           "output",
                                           "Output Gaussian-corrupted image",
                                           true,
                                           "",
                                           "Output Gaussian-corrupted image",
                                           cmd);

    TCLAP::ValueArg<std::string> locationArg("l",
                                           "location",
                                           "Output location image",
                                           false,
                                           "",
                                           "Output location image",
                                           cmd);

    TCLAP::ValueArg<std::string> scaleArg("s",
                                           "scale",
                                           "Output scale image",
                                           false,
                                           "",
                                           "Output scale image",
                                           cmd);

    TCLAP::ValueArg<std::string> maskArg("m",
                                           "mask",
                                           "Optional segmentation mask",
                                           false,
                                           "",
                                           "Optional segmentation mask",
                                           cmd);

    TCLAP::ValueArg<double> epsArg("E",
                                           "epsilon",
                                           "Minimal absolute value difference betweem fixed point iterations (default: 1e-8)",
                                           false,
                                           1.0e-8,
                                           "Minimal absolute value difference betweem fixed point iterations",
                                           cmd);

    TCLAP::ValueArg<double> sigmaArg("S",
                                           "sigma",
                                           "Gaussian standard deviation for defining neighbor weights (default: 1.0)",
                                           false,
                                           1.0,
                                           "Gaussian standard deviation for defining neighbor weights",
                                           cmd);

    TCLAP::ValueArg<unsigned int> radiusArg("R",
                                               "radius",
                                               "Neighborhood radius (default: 1)",
                                               false,
                                               1,
                                               "Neighborhood radius",
                                               cmd);

    TCLAP::ValueArg<unsigned int> maxiterArg("I",
                                               "maxiter",
                                               "Maximum number of iterations (default: 100)",
                                               false,
                                               100,
                                               "Maximum number of iterations",
                                               cmd);

    TCLAP::ValueArg<unsigned int> nbpArg("T",
                                         "nbp",
                                         "Number of threads to run on -> default : automatically determine",
                                         false,
                                         itk::MultiThreader::GetGlobalDefaultNumberOfThreads(),
                                         "Number of threads",
                                         cmd);

    try
    {
        cmd.parse(ac,av);
    }
    catch (TCLAP::ArgException& e)
    {
        std::cerr << "Error: " << e.error() << "for argument " << e.argId() << std::endl;
        return EXIT_FAILURE;
    }

    const unsigned int ImageDimension = 3;
    typedef anima::RiceToGaussianImageFilter<ImageDimension> RiceToGaussianImageFilterType;
    typedef RiceToGaussianImageFilterType::InputImageType InputImageType;
    typedef RiceToGaussianImageFilterType::OutputImageType OutputImageType;
    typedef RiceToGaussianImageFilterType::MaskImageType MaskImageType;

    RiceToGaussianImageFilterType::Pointer filter = RiceToGaussianImageFilterType::New();
    filter->SetInput(anima::readImage<InputImageType>(inputArg.getValue()));
    filter->SetRadius(radiusArg.getValue());
    filter->SetMaximumNumberOfIterations(maxiterArg.getValue());
    filter->SetEpsilon(epsArg.getValue());
    filter->SetSigma(sigmaArg.getValue());
    filter->SetNumberOfThreads(nbpArg.getValue());

    if (maskArg.getValue() != "")
      filter->SetSegmentationMask(anima::readImage<MaskImageType>(maskArg.getValue()));

    itk::CStyleCommand::Pointer callback = itk::CStyleCommand::New();
    callback->SetCallback(eventCallback);
    filter->AddObserver(itk::ProgressEvent(), callback);

    std::cout << "Running Gaussianizer..." << std::endl;

    try
    {
      filter->Update();
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "\nDone. Writing result to disk..." << std::endl;

    anima::writeImage<OutputImageType>(outputArg.getValue(), filter->GetGaussianImage());

    if (locationArg.getValue() != "")      
      anima::writeImage<OutputImageType>(locationArg.getValue(), filter->GetLocationImage());

    if (scaleArg.getValue() != "")
      anima::writeImage<OutputImageType>(scaleArg.getValue(), filter->GetScaleImage());
    
    return EXIT_SUCCESS;
}
