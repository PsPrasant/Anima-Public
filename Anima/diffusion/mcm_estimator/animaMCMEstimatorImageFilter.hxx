#pragma once
#include "animaMCMEstimatorImageFilter.h"

#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>

#include <animaNLOPTOptimizers.h>
#include <itkLevenbergMarquardtOptimizer.h>

#include <animaGaussianMCMCostFunction.h>
#include <animaGaussianMCMVariableProjectionSingleValuedCostFunction.h>
#include <animaGaussianMCMVariableProjectionMultipleValuedCostFunction.h>

#include <animaVectorOperations.h>

#include <animaDTIEstimationImageFilter.h>
#include <animaBaseTensorTools.h>

#include <animaTensorCompartment.h>
#include <animaMCMFileWriter.h>

#include <boost/math/tools/toms748_solve.hpp>
#include <ctime>

namespace anima
{

template <class InputPixelType, class OutputPixelType>
void
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>
::AddGradientDirection(unsigned int i, GradientType &grad)
{
    if (i == m_GradientDirections.size())
        m_GradientDirections.push_back(grad);
    else if (i > m_GradientDirections.size())
        std::cerr << "Trying to add a direction not contiguous... Add directions contiguously (0,1,2,3,...)..." << std::endl;
    else
        m_GradientDirections[i] = grad;
}

template <class InputPixelType, class OutputPixelType>
typename MCMEstimatorImageFilter<InputPixelType, OutputPixelType>::MCMCreatorType *
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>
::GetNewMCMCreatorInstance()
{
    return new MCMCreatorType;
}

template <class InputPixelType, class OutputPixelType>
void
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>
::GenerateOutputInformation()
{
    // Override the method in itkImageSource, so we can set the vector length of
    // the output itk::VectorImage

    this->Superclass::GenerateOutputInformation();

    OutputImageType *output = this->GetOutput();

    // Create fake MCM output to get its length
    MCMCreatorType *tmpMCMCreator = this->GetNewMCMCreatorInstance();
    tmpMCMCreator->SetModelWithFreeWaterComponent(m_ModelWithFreeWaterComponent);
    tmpMCMCreator->SetModelWithStationaryWaterComponent(m_ModelWithStationaryWaterComponent);
    tmpMCMCreator->SetModelWithRestrictedWaterComponent(m_ModelWithRestrictedWaterComponent);
    tmpMCMCreator->SetCompartmentType(m_CompartmentType);
    tmpMCMCreator->SetNumberOfCompartments(m_NumberOfCompartments);

    MCMPointer tmpMCM = tmpMCMCreator->GetNewMultiCompartmentModel();

    output->SetVectorLength(tmpMCM->GetSize());
    output->SetDescriptionModel(tmpMCM);

    delete tmpMCMCreator;
}

template <class InputPixelType, class OutputPixelType>
void
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>
::WriteMCMOutput(std::string fileName)
{
    MCMPointer tmpMCM = m_MCMCreators[0]->GetNewMultiCompartmentModel();

    typedef anima::MCMFileWriter <OutputPixelType, InputImageType::ImageDimension> MCMFileWriterType;
    MCMFileWriterType writer;

    writer.SetInputImage(this->GetOutput());
    writer.SetFileName(fileName);

    writer.Update();
}

template <class InputPixelType, class OutputPixelType>
void
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>
::CheckComputationMask()
{
    typedef itk::ImageRegionConstIterator <InputImageType> B0IteratorType;
    typedef itk::ImageRegionIterator <MaskImageType> MaskIteratorType;

    unsigned int firstB0Index = 0;
    while (m_BValuesList[firstB0Index] > 10)
        ++firstB0Index;

    B0IteratorType b0Itr(this->GetInput(firstB0Index),this->GetOutput()->GetLargestPossibleRegion());

    if (!this->GetComputationMask())
        this->Superclass::CheckComputationMask();

    MaskIteratorType maskItr(this->GetComputationMask(),this->GetOutput()->GetLargestPossibleRegion());

    while (!b0Itr.IsAtEnd())
    {
        if ((maskItr.Get() != 0)&&(b0Itr.Get() <= m_B0Threshold))
            maskItr.Set(0);

        ++b0Itr;
        ++maskItr;
    }
}

template <class InputPixelType, class OutputPixelType>
void
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>
::BeforeThreadedGenerateData()
{
    if ((m_Optimizer == "levenberg")&&((m_MLEstimationStrategy != VariableProjection)||(m_NoiseType != Gaussian)))
        itkExceptionMacro("Levenberg Marquardt optimizer only working with Gaussian noise and variable projection");

    if ((m_Optimizer != "bobyqa")&&m_UseCommonDiffusivities)
        itkExceptionMacro("Derivative based optimizers not supported yet for common parameters, use Bobyqa instead");

    if ((m_NoiseType == NCC)&&(m_MLEstimationStrategy != Profile))
        itkExceptionMacro("NCC noise is only compatible with profile estimation strategy");

    srand(time(0));

    m_NumberOfImages = this->GetNumberOfIndexedInputs();

    if (m_BValuesList.size() != m_NumberOfImages)
        itkExceptionMacro("There should be the same number of input images and input b-values...");

    itk::ImageRegionIterator <OutputImageType> fillOut(this->GetOutput(),this->GetOutput()->GetLargestPossibleRegion());
    unsigned int outSize = this->GetOutput()->GetNumberOfComponentsPerPixel();
    typename OutputImageType::PixelType emptyModelVec(outSize);
    emptyModelVec.Fill(0);

    while (!fillOut.IsAtEnd())
    {
        fillOut.Set(emptyModelVec);
        ++fillOut;
    }

    // Create AICc volume
    m_AICcVolume = OutputScalarImageType::New();
    m_AICcVolume->Initialize();
    m_AICcVolume->SetRegions(this->GetInput(0)->GetLargestPossibleRegion());
    m_AICcVolume->SetSpacing (this->GetInput(0)->GetSpacing());
    m_AICcVolume->SetOrigin (this->GetInput(0)->GetOrigin());
    m_AICcVolume->SetDirection (this->GetInput(0)->GetDirection());
    m_AICcVolume->Allocate();
    m_AICcVolume->FillBuffer(0);

    // Create B0 volume
    m_B0Volume = OutputScalarImageType::New();
    m_B0Volume->Initialize();
    m_B0Volume->SetRegions(this->GetInput(0)->GetLargestPossibleRegion());
    m_B0Volume->SetSpacing (this->GetInput(0)->GetSpacing());
    m_B0Volume->SetOrigin (this->GetInput(0)->GetOrigin());
    m_B0Volume->SetDirection (this->GetInput(0)->GetDirection());
    m_B0Volume->Allocate();
    m_B0Volume->FillBuffer(0);

    // Create sigma volume
    m_SigmaSquareVolume = OutputScalarImageType::New();
    m_SigmaSquareVolume->Initialize();
    m_SigmaSquareVolume->SetRegions(this->GetInput(0)->GetLargestPossibleRegion());
    m_SigmaSquareVolume->SetSpacing (this->GetInput(0)->GetSpacing());
    m_SigmaSquareVolume->SetOrigin (this->GetInput(0)->GetOrigin());
    m_SigmaSquareVolume->SetDirection (this->GetInput(0)->GetDirection());
    m_SigmaSquareVolume->Allocate();
    m_SigmaSquareVolume->FillBuffer(0);

    // Create mose volume
    if (!m_MoseVolume)
    {
        m_MoseVolume = MoseImageType::New();
        m_MoseVolume->Initialize();
        m_MoseVolume->SetRegions(this->GetInput(0)->GetLargestPossibleRegion());
        m_MoseVolume->SetSpacing (this->GetInput(0)->GetSpacing());
        m_MoseVolume->SetOrigin (this->GetInput(0)->GetOrigin());
        m_MoseVolume->SetDirection (this->GetInput(0)->GetDirection());
        m_MoseVolume->Allocate();
        m_MoseVolume->FillBuffer(0);
    }

    if (m_ExternalMoseVolume)
        m_FindOptimalNumberOfCompartments = false;

    Superclass::BeforeThreadedGenerateData();

    m_MCMCreators.resize(this->GetNumberOfThreads());
    for (unsigned int i = 0;i < this->GetNumberOfThreads();++i)
        m_MCMCreators[i] = this->GetNewMCMCreatorInstance();

    typedef anima::DTIEstimationImageFilter <InputPixelType, OutputPixelType> DTIEstimationFilterType;
    typename DTIEstimationFilterType::Pointer dtiEstimator = DTIEstimationFilterType::New();

    dtiEstimator->SetBValuesList(m_BValuesList);
    dtiEstimator->SetNumberOfThreads(this->GetNumberOfThreads());
    dtiEstimator->SetComputationMask(this->GetComputationMask());
    dtiEstimator->SetVerboseProgression(false);

    for(unsigned int i = 0;i < this->GetNumberOfIndexedInputs();++i)
    {
        dtiEstimator->AddGradientDirection(i,m_GradientDirections[i]);
        dtiEstimator->SetInput(i,this->GetInput(i));
    }

    dtiEstimator->Update();

    m_InitialDTImage = dtiEstimator->GetOutput();
    m_InitialDTImage->DisconnectPipeline();

    if (!m_ExternalDTIParameters)
    {
        m_AxialDiffusivityFixedValue = 0;
        m_RadialDiffusivity1FixedValue = 0;
        m_RadialDiffusivity2FixedValue = 0;
        unsigned int numValues = 0;

        typedef itk::ImageRegionConstIterator <VectorImageType> DTImageIteratorType;
        unsigned int tensDim = 3;
        vnl_matrix <double> tmpTensor(tensDim,tensDim);

        typedef itk::SymmetricEigenAnalysis < vnl_matrix <double>, vnl_diag_matrix<double>, vnl_matrix <double> > EigenAnalysisType;

        EigenAnalysisType eigenAnalysis(tensDim);
        vnl_diag_matrix <double> eigVals(tensDim);

        VariableLengthVectorType dataDTI(6);
        DTImageIteratorType dtiIterator (m_InitialDTImage,m_InitialDTImage->GetLargestPossibleRegion());
        while (!dtiIterator.IsAtEnd())
        {
            dataDTI = dtiIterator.Get();

            bool zeroTensor = true;
            for (unsigned int i = 0;i < 6;++i)
            {
                if (dataDTI[i] != 0)
                {
                    zeroTensor = false;
                    break;
                }
            }

            if (zeroTensor)
            {
                ++dtiIterator;
                continue;
            }

            anima::GetTensorFromVectorRepresentation(dataDTI,tmpTensor,tensDim);
            eigenAnalysis.ComputeEigenValues(tmpTensor,eigVals);

            if (eigVals[0] <= 0)
            {
                ++dtiIterator;
                continue;
            }

            double fa = 0;
            double adc = 0;
            double faDenom = 0;

            for (unsigned int i = 0;i < tensDim;++i)
            {
                faDenom += eigVals[i] * eigVals[i];
                adc += eigVals[i];
                for (unsigned int j = i+1;j < tensDim;++j)
                    fa += (eigVals[i] - eigVals[j])*(eigVals[i] - eigVals[j]);
            }

            fa = std::sqrt(fa / (2.0 * faDenom));
            adc /= tensDim;

            if ((fa > 0.8) && (adc < 3.0e-3) && (adc > 1.0e-4))
            {
                m_AxialDiffusivityFixedValue += eigVals[2];
                m_RadialDiffusivity1FixedValue += eigVals[1];
                m_RadialDiffusivity2FixedValue += eigVals[0];
                ++numValues;
            }

            ++dtiIterator;
        }

        if (numValues > 0)
        {
            m_AxialDiffusivityFixedValue /= numValues;
            m_RadialDiffusivity1FixedValue /= numValues;
            m_RadialDiffusivity2FixedValue /= numValues;
        }
        else
        {
            m_AxialDiffusivityFixedValue = 1.7e-3;
            m_RadialDiffusivity1FixedValue = 1.5e-4;
            m_RadialDiffusivity2FixedValue = 1.5e-4;
        }
    }
    
    std::cout << "Stick diffusivities derived from DTI:" << std::endl;
    std::cout << " - Axial diffusivity: " << m_AxialDiffusivityFixedValue << " mm2/s," << std::endl;
    std::cout << " - First radial diffusivity: " << m_RadialDiffusivity1FixedValue << " mm2/s," << std::endl;
    std::cout << " - Second radial diffusivity: " << m_RadialDiffusivity2FixedValue << " mm2/s." << std::endl;

    // Setting up creators
    if (m_Optimizer == "levenberg")
        m_UseBoundedOptimization = true;

    for (unsigned int i = 0;i < this->GetNumberOfThreads();++i)
    {
        m_MCMCreators[i]->SetAxialDiffusivityValue(m_AxialDiffusivityFixedValue);
        m_MCMCreators[i]->SetFreeWaterDiffusivityValue(3.0e-3);
        m_MCMCreators[i]->SetRadialDiffusivity1Value(m_RadialDiffusivity1FixedValue);
        m_MCMCreators[i]->SetRadialDiffusivity2Value(m_RadialDiffusivity2FixedValue);
        m_MCMCreators[i]->SetUseBoundedOptimization(m_UseBoundedOptimization);
    }

    if (m_UseConcentrationBoundsFromDTI)
    {
        // Orientation concentration bounds estimation
        ConcentrationUpperBoundSolverCostFunction upperCost;
        upperCost.SetWMAxialDiffusivity(m_AxialDiffusivityFixedValue);
        upperCost.SetWMRadialDiffusivity((m_RadialDiffusivity1FixedValue + m_RadialDiffusivity2FixedValue) / 2.0);

        boost::uintmax_t max_iter = 500;
        boost::math::tools::eps_tolerance<double> tol(30);

        double kappaLowerBound = m_AxialDiffusivityFixedValue * 2.0 / (m_RadialDiffusivity1FixedValue + m_RadialDiffusivity2FixedValue) - 1.0;

        std::pair <double,double> r = boost::math::tools::toms748_solve(upperCost, kappaLowerBound, 20.0, tol, max_iter);
        double kappaUpperBound = std::min(r.first,r.second);

        for (unsigned int i = 0;i < this->GetNumberOfThreads();++i)
            m_MCMCreators[i]->SetConcentrationBounds(kappaLowerBound,kappaUpperBound);
    }
}

template <class InputPixelType, class OutputPixelType>
void
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>
::ThreadedGenerateData(const OutputImageRegionType &outputRegionForThread, itk::ThreadIdType threadId)
{
    typedef itk::ImageRegionConstIterator <InputImageType> ConstImageIteratorType;

    std::vector <ConstImageIteratorType> inIterators(m_NumberOfImages);
    for (unsigned int i = 0;i < m_NumberOfImages;++i)
        inIterators[i] = ConstImageIteratorType(this->GetInput(i),outputRegionForThread);

    typedef itk::ImageRegionIterator <OutputImageType> OutImageIteratorType;
    typedef itk::ImageRegionConstIterator <VectorImageType> VectorImageIteratorType;
    OutImageIteratorType outIterator(this->GetOutput(),outputRegionForThread);
    VectorImageIteratorType initDTIterator(m_InitialDTImage,outputRegionForThread);

    typedef itk::ImageRegionIterator <MaskImageType> MaskIteratorType;
    MaskIteratorType maskItr(this->GetComputationMask(),outputRegionForThread);

    typedef itk::ImageRegionIterator <OutputScalarImageType> ImageIteratorType;
    ImageIteratorType aiccIterator(m_AICcVolume, outputRegionForThread);
    ImageIteratorType b0Iterator(m_B0Volume, outputRegionForThread);
    ImageIteratorType sigmaIterator(m_SigmaSquareVolume, outputRegionForThread);

    typedef itk::ImageRegionIterator <MoseImageType> MoseIteratorType;
    MoseIteratorType moseIterator(m_MoseVolume, outputRegionForThread);

    std::vector <double> observedSignals(m_NumberOfImages,0);

    typename OutputImageType::PixelType resVec(this->GetOutput()->GetNumberOfComponentsPerPixel());

    MCMPointer mcmData = 0, mcmValue = 0;
    MCMPointer outputMCMData = this->GetOutput()->GetDescriptionModel()->Clone();
    MCMType::ListType outputWeights(outputMCMData->GetNumberOfCompartments(),0);

    double aiccValue, b0Value, sigmaSqValue;
    anima::BaseCompartment::ModelOutputVectorType initDTIValue;

    while (!outIterator.IsAtEnd())
    {
        resVec.Fill(0.0);

        if (maskItr.Get() == 0)
        {
            outIterator.Set(resVec);

            for (unsigned int i = 0;i < m_NumberOfImages;++i)
                ++inIterators[i];

            ++outIterator;
            ++maskItr;
            ++aiccIterator;
            ++b0Iterator;
            ++sigmaIterator;
            ++initDTIterator;
            ++moseIterator;

            continue;
        }

        // Load DWI
        for (unsigned int i = 0;i < m_NumberOfImages;++i)
            observedSignals[i] = inIterators[i].Get();
        
        int moseValue = -1;
        bool estimateNonIsoCompartments = false;
        if (m_ExternalMoseVolume)
        {
            moseValue = moseIterator.Get();
            if (moseValue > 0)
                estimateNonIsoCompartments = true;
        }
        else if (m_NumberOfCompartments > 0)
            estimateNonIsoCompartments = true;

        if (estimateNonIsoCompartments)
        {
            initDTIValue = initDTIterator.Get();

            // If model selection, handle it here
            unsigned int minimalNumberOfCompartments = m_NumberOfCompartments;
            unsigned int maximalNumberOfCompartments = m_NumberOfCompartments;
            if (m_FindOptimalNumberOfCompartments)
            {
                minimalNumberOfCompartments = 1;
                moseValue = 0;

                if (m_ModelWithFreeWaterComponent || m_ModelWithRestrictedWaterComponent || m_ModelWithStationaryWaterComponent)
                    this->EstimateFreeWaterModel(mcmData,observedSignals,threadId,aiccValue,b0Value,sigmaSqValue);
            }
            else if (moseValue != -1)
            {
                minimalNumberOfCompartments = moseValue;
                maximalNumberOfCompartments = moseValue;
            }

            for (unsigned int i = minimalNumberOfCompartments;i <= maximalNumberOfCompartments;++i)
            {
                double tmpB0Value = 0;
                double tmpSigmaSqValue = 0;
                double tmpAiccValue = 0;
                MCMPointer mcmValue;

                this->OptimizeNonIsotropicCompartments(mcmValue,i,initDTIValue,observedSignals,threadId,tmpAiccValue,tmpB0Value,tmpSigmaSqValue);

                if ((tmpAiccValue < aiccValue)||(!m_FindOptimalNumberOfCompartments))
                {
                    aiccValue = tmpAiccValue;
                    if (m_FindOptimalNumberOfCompartments)
                        moseValue = i;
                    b0Value = tmpB0Value;
                    sigmaSqValue = tmpSigmaSqValue;
                    mcmData = mcmValue;
                }
            }
        }
        else if (m_ModelWithFreeWaterComponent || m_ModelWithRestrictedWaterComponent || m_ModelWithStationaryWaterComponent)
            this->EstimateFreeWaterModel(mcmData,observedSignals,threadId,aiccValue,b0Value,sigmaSqValue);
        else
            itkExceptionMacro("Nothing to estimate...");

        if (outputMCMData->GetNumberOfCompartments() != mcmData->GetNumberOfCompartments())
        {
            // If we are selecting the number of compartments, create some empty ones here
            std::fill(outputWeights.begin(),outputWeights.end(),0.0);
            for (unsigned int i = 0;i < mcmData->GetNumberOfCompartments();++i)
            {
                outputMCMData->GetCompartment(i)->CopyFromOther(mcmData->GetCompartment(i));
                outputWeights[i] = mcmData->GetCompartmentWeight(i);
            }

            outputMCMData->SetCompartmentWeights(outputWeights);
            resVec = outputMCMData->GetModelVector();
        }
        else
            resVec = mcmData->GetModelVector();

        outIterator.Set(resVec);
        aiccIterator.Set(aiccValue);
        b0Iterator.Set(b0Value);
        sigmaIterator.Set(sigmaSqValue);
        moseIterator.Set(mcmData->GetNumberOfCompartments() - mcmData->GetNumberOfIsotropicCompartments());

        for (unsigned int i = 0;i < m_NumberOfImages;++i)
            ++inIterators[i];

        ++outIterator;
        ++maskItr;
        ++aiccIterator;
        ++b0Iterator;
        ++sigmaIterator;
        ++initDTIterator;
        ++moseIterator;
    }
}

template <class InputPixelType, class OutputPixelType>
void
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>
::OptimizeNonIsotropicCompartments(MCMPointer &mcmValue, unsigned int currentNumberOfCompartments,
                                   BaseCompartment::ModelOutputVectorType &initialDTI,
                                   std::vector <double> &observedSignals, itk::ThreadIdType threadId,
                                   double &aiccValue, double &b0Value, double &sigmaSqValue)
{
    b0Value = 0;
    sigmaSqValue = 1;
    aiccValue = -1;

    double optimalB0Value = 0;
    double optimalSigmaSqValue = 0;
    double optimalAiccValue = 0;

    std::vector <double> samplingLowerBounds;
    std::vector <double> samplingUpperBounds;

    unsigned int restartTotalNumber = 1;
    if (currentNumberOfCompartments > 1)
        restartTotalNumber = m_NumberOfRandomRestarts;

    SequenceGeneratorType generator(3*currentNumberOfCompartments);
    samplingLowerBounds.resize(3*currentNumberOfCompartments);
    std::fill(samplingLowerBounds.begin(),samplingLowerBounds.end(),0.0);
    generator.SetLowerBounds(samplingLowerBounds);

    samplingUpperBounds.resize(3*currentNumberOfCompartments);
    for (unsigned int j = 0;j < currentNumberOfCompartments;++j)
    {
        samplingUpperBounds[j] = 1.0;
        samplingUpperBounds[currentNumberOfCompartments+j] = M_PI;
        samplingUpperBounds[2*currentNumberOfCompartments+j] = 2.0 * M_PI;
    }

    generator.SetUpperBounds(samplingUpperBounds);
    MCMPointer mcmOptimizationValue;
    for (unsigned int restartNum = 0;restartNum < restartTotalNumber;++restartNum)
    {
        this->InitialOrientationsEstimation(mcmOptimizationValue,currentNumberOfCompartments,initialDTI,observedSignals,
                                            generator,threadId,aiccValue,b0Value,sigmaSqValue);

        this->TrunkModelEstimation(mcmOptimizationValue,observedSignals,threadId,aiccValue,b0Value,sigmaSqValue);
        if ((m_CompartmentType != Stick)&&(m_CompartmentType != Zeppelin))
            this->SpecificModelEstimation(mcmOptimizationValue,observedSignals,threadId,aiccValue,b0Value,sigmaSqValue);

        if ((aiccValue < optimalAiccValue)||(restartNum == 0))
        {
            optimalB0Value = b0Value;
            optimalAiccValue = aiccValue;
            optimalSigmaSqValue = sigmaSqValue;
            mcmValue = mcmOptimizationValue;
        }
    }

    aiccValue = optimalAiccValue;
    sigmaSqValue = optimalSigmaSqValue;
    b0Value = optimalB0Value;
}

template <class InputPixelType, class OutputPixelType>
void
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>
::EstimateFreeWaterModel(MCMPointer &mcmValue, std::vector <double> &observedSignals, itk::ThreadIdType threadId,
                         double &aiccValue, double &b0Value, double &sigmaSqValue)
{
    // Declarations for optimization
    MCMCreatorType *mcmCreator = m_MCMCreators[threadId];
    mcmCreator->SetModelWithFreeWaterComponent(m_ModelWithFreeWaterComponent);
    mcmCreator->SetModelWithStationaryWaterComponent(m_ModelWithStationaryWaterComponent);
    mcmCreator->SetModelWithRestrictedWaterComponent(m_ModelWithRestrictedWaterComponent);
    mcmCreator->SetNumberOfCompartments(0);
    mcmCreator->SetUseFixedWeights(m_UseFixedWeights || (m_MLEstimationStrategy == VariableProjection));
    mcmCreator->SetUseConstrainedFreeWaterDiffusivity(m_UseConstrainedFreeWaterDiffusivity);
    mcmCreator->SetUseConstrainedIRWDiffusivity(m_UseConstrainedIRWDiffusivity);
    mcmCreator->SetFreeWaterProportionFixedValue(m_FreeWaterProportionFixedValue);
    mcmCreator->SetStationaryWaterProportionFixedValue(m_StationaryWaterProportionFixedValue);
    mcmCreator->SetRestrictedWaterProportionFixedValue(m_RestrictedWaterProportionFixedValue);

    mcmValue = mcmCreator->GetNewMultiCompartmentModel();

    b0Value = 0;
    sigmaSqValue = 1;

    CostFunctionBasePointer cost = this->CreateCostFunction(observedSignals,mcmValue);

    unsigned int dimension = mcmValue->GetNumberOfParameters();
    ParametersType p(dimension);
    MCMType::ListType workVec(dimension);

    if (dimension > 0)
    {
        itk::Array<double> lowerBounds(dimension), upperBounds(dimension);

        workVec = mcmValue->GetParameterLowerBounds();
        for (unsigned int i = 0;i < dimension;++i)
            lowerBounds[i] = workVec[i];

        workVec = mcmValue->GetParameterUpperBounds();
        for (unsigned int i = 0;i < dimension;++i)
            upperBounds[i] = workVec[i];

        workVec = mcmValue->GetParametersAsVector();
        for (unsigned int i = 0;i < dimension;++i)
            p[i] = workVec[i];

        double costValue = this->PerformSingleOptimization(p,cost,lowerBounds,upperBounds);

        // - Get estimated DTI and B0
        for (unsigned int i = 0;i < dimension;++i)
            workVec[i] = p[i];

        mcmValue->SetParametersFromVector(workVec);

        this->GetProfiledInformation(cost,mcmValue,b0Value,sigmaSqValue);
        aiccValue = this->ComputeAICcValue(mcmValue,costValue);
    }
    else
    {
        double costValue = this->GetCostValue(cost,p);
        this->GetProfiledInformation(cost,mcmValue,b0Value,sigmaSqValue);

        aiccValue = this->ComputeAICcValue(mcmValue,costValue);
    }
}

template <class InputPixelType, class OutputPixelType>
typename MCMEstimatorImageFilter<InputPixelType, OutputPixelType>::CostFunctionBasePointer
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>::CreateCostFunction(std::vector <double> &observedSignals, MCMPointer &mcmModel)
{
    CostFunctionBasePointer returnCost;

    if (m_NoiseType == Gaussian)
    {
        if (m_MLEstimationStrategy != VariableProjection)
        {
            anima::GaussianMCMCostFunction::Pointer tmpCost = anima::GaussianMCMCostFunction::New();
            tmpCost->SetMarginalEstimation(m_MLEstimationStrategy == Marginal);
            tmpCost->SetObservedSignals(observedSignals);
            tmpCost->SetGradients(m_GradientDirections);
            tmpCost->SetBValues(m_BValuesList);
            tmpCost->SetMCMStructure(mcmModel);

            returnCost = tmpCost;
        }
        else
        {
            anima::GaussianMCMVariableProjectionCost::Pointer internalCost = anima::GaussianMCMVariableProjectionCost::New();
            internalCost->SetObservedSignals(observedSignals);
            internalCost->SetGradients(m_GradientDirections);
            internalCost->SetBValues(m_BValuesList);
            internalCost->SetMCMStructure(mcmModel);

            if (m_Optimizer == "levenberg")
            {
                anima::GaussianMCMVariableProjectionMultipleValuedCostFunction::Pointer tmpCost =
                        anima::GaussianMCMVariableProjectionMultipleValuedCostFunction::New();

                internalCost->SetUseDerivative(!m_VNLDerivativeComputation);
                tmpCost->SetInternalCost(internalCost);

                returnCost = tmpCost;
            }
            else
            {
                anima::GaussianMCMVariableProjectionSingleValuedCostFunction::Pointer tmpCost =
                        anima::GaussianMCMVariableProjectionSingleValuedCostFunction::New();

                internalCost->SetUseDerivative(m_Optimizer == "ccsaq");
                tmpCost->SetInternalCost(internalCost);

                returnCost = tmpCost;
            }
        }
    }
    else
        itkExceptionMacro("Cost function type not supported")

    return returnCost;
}

template <class InputPixelType, class OutputPixelType>
void
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>
::InitialOrientationsEstimation(MCMPointer &mcmValue, unsigned int currentNumberOfCompartments,
                                BaseCompartment::ModelOutputVectorType &initialDTI,
                                std::vector <double> &observedSignals, SequenceGeneratorType &generator,
                                itk::ThreadIdType threadId, double &aiccValue, double &b0Value, double &sigmaSqValue)
{
    // Single DTI is already estimated, get it to use in next step
    // - First create output object
    MCMCreatorType *mcmCreator = m_MCMCreators[threadId];
    mcmCreator->SetModelWithFreeWaterComponent(false);
    mcmCreator->SetModelWithStationaryWaterComponent(false);
    mcmCreator->SetModelWithRestrictedWaterComponent(false);
    mcmCreator->SetCompartmentType(Tensor);
    mcmCreator->SetNumberOfCompartments(1);
    mcmCreator->SetUseFixedWeights(true);
    mcmCreator->SetUseConstrainedFreeWaterDiffusivity(false);
    mcmCreator->SetUseConstrainedIRWDiffusivity(false);
    mcmCreator->SetUseConstrainedDiffusivity(false);

    MCMPointer mcmDTIValue = mcmCreator->GetNewMultiCompartmentModel();
    mcmDTIValue->GetCompartment(0)->SetCompartmentVector(initialDTI);

    b0Value = 0;
    sigmaSqValue = 1;
    aiccValue = -1;

    // - First create model
    mcmCreator->SetModelWithFreeWaterComponent(m_ModelWithFreeWaterComponent);
    mcmCreator->SetModelWithStationaryWaterComponent(m_ModelWithStationaryWaterComponent);
    mcmCreator->SetModelWithRestrictedWaterComponent(m_ModelWithRestrictedWaterComponent);
    mcmCreator->SetCompartmentType(Stick);
    mcmCreator->SetNumberOfCompartments(currentNumberOfCompartments);
    mcmCreator->SetUseFixedWeights(m_UseFixedWeights || (m_MLEstimationStrategy == VariableProjection));
    mcmCreator->SetFreeWaterProportionFixedValue(m_FreeWaterProportionFixedValue);
    mcmCreator->SetStationaryWaterProportionFixedValue(m_StationaryWaterProportionFixedValue);
    mcmCreator->SetRestrictedWaterProportionFixedValue(m_RestrictedWaterProportionFixedValue);
    mcmCreator->SetUseConstrainedDiffusivity(true);
    mcmCreator->SetUseConstrainedFreeWaterDiffusivity(m_UseConstrainedFreeWaterDiffusivity);
    mcmCreator->SetUseConstrainedIRWDiffusivity(m_UseConstrainedIRWDiffusivity);
    mcmCreator->SetUseCommonDiffusivities(m_UseCommonDiffusivities);

    MCMPointer mcmUpdateValue = mcmCreator->GetNewMultiCompartmentModel();

    unsigned int dimension = mcmUpdateValue->GetNumberOfParameters();
    ParametersType p(dimension);
    MCMType::ListType workVec(dimension);
    itk::Array<double> lowerBounds(dimension), upperBounds(dimension);

    workVec = mcmUpdateValue->GetParameterLowerBounds();
    for (unsigned int j = 0;j < dimension;++j)
        lowerBounds[j] = workVec[j];

    workVec = mcmUpdateValue->GetParameterUpperBounds();
    for (unsigned int j = 0;j < dimension;++j)
        upperBounds[j] = workVec[j];

    CostFunctionBasePointer cost = this->CreateCostFunction(observedSignals,mcmUpdateValue);

    // - Now the tricky part: initialize from previous model, handled somewhere else
    this->InitializeStickModelFromDTI(mcmDTIValue,mcmUpdateValue,generator);

    // - Update ball and stick model against observed signals
    workVec = mcmUpdateValue->GetParametersAsVector();
    for (unsigned int j = 0;j < dimension;++j)
        p[j] = workVec[j];

    double costValue = this->PerformSingleOptimization(p,cost,lowerBounds,upperBounds);

    // - Get estimated data
    for (unsigned int j = 0;j < dimension;++j)
        workVec[j] = p[j];

    mcmUpdateValue->SetParametersFromVector(workVec);

    this->GetProfiledInformation(cost,mcmUpdateValue,b0Value,sigmaSqValue);

    aiccValue = this->ComputeAICcValue(mcmUpdateValue,costValue);
    mcmValue = mcmUpdateValue;
}

template <class InputPixelType, class OutputPixelType>
void
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>
::TrunkModelEstimation(MCMPointer &mcmValue, std::vector <double> &observedSignals, itk::ThreadIdType threadId,
                       double &aiccValue, double &b0Value, double &sigmaSqValue)
{
    unsigned int optimalNumberOfCompartments = mcmValue->GetNumberOfCompartments() - mcmValue->GetNumberOfIsotropicCompartments();

    //Already done in initial orientations estimation
    if ((m_CompartmentType == Stick)&&(m_UseConstrainedDiffusivity))
        return;

    // - First create model
    MCMCreatorType *mcmCreator = m_MCMCreators[threadId];
    mcmCreator->SetCompartmentType(Stick);
    mcmCreator->SetNumberOfCompartments(optimalNumberOfCompartments);
    mcmCreator->SetUseConstrainedDiffusivity(m_UseConstrainedDiffusivity);

    MCMPointer mcmUpdateValue = mcmCreator->GetNewMultiCompartmentModel();

    // - Now the tricky part: initialize from previous model, handled somewhere else
    this->InitializeModelFromSimplifiedOne(mcmValue,mcmUpdateValue);

    CostFunctionBasePointer cost = this->CreateCostFunction(observedSignals,mcmUpdateValue);

    unsigned int dimension = mcmUpdateValue->GetNumberOfParameters();
    ParametersType p(dimension);
    MCMType::ListType workVec(dimension);
    itk::Array<double> lowerBounds(dimension), upperBounds(dimension);

    workVec = mcmUpdateValue->GetParameterLowerBounds();
    for (unsigned int i = 0;i < dimension;++i)
        lowerBounds[i] = workVec[i];

    workVec = mcmUpdateValue->GetParameterUpperBounds();
    for (unsigned int i = 0;i < dimension;++i)
        upperBounds[i] = workVec[i];

    workVec = mcmUpdateValue->GetParametersAsVector();
    for (unsigned int i = 0;i < dimension;++i)
        p[i] = workVec[i];
    
    double costValue = this->PerformSingleOptimization(p,cost,lowerBounds,upperBounds);
    this->GetProfiledInformation(cost,mcmUpdateValue,b0Value,sigmaSqValue);
    
    for (unsigned int i = 0;i < dimension;++i)
        workVec[i] = p[i];

    mcmUpdateValue->SetParametersFromVector(workVec);
    
    mcmValue = mcmUpdateValue;
    aiccValue = this->ComputeAICcValue(mcmValue,costValue);

    if (m_CompartmentType == Stick)
        return;
    
    // We're done with ball and stick, next up is ball and zeppelin
    // - First create model
    mcmCreator->SetCompartmentType(Zeppelin);
    mcmUpdateValue = mcmCreator->GetNewMultiCompartmentModel();

    // - Now the tricky part: initialize from previous model, handled somewhere else
    this->InitializeModelFromSimplifiedOne(mcmValue,mcmUpdateValue);

    // - Update ball and zeppelin model against observed signals
    cost = this->CreateCostFunction(observedSignals,mcmUpdateValue);
    dimension = mcmUpdateValue->GetNumberOfParameters();
    p.SetSize(dimension);
    lowerBounds.SetSize(dimension);
    upperBounds.SetSize(dimension);

    workVec = mcmUpdateValue->GetParameterLowerBounds();
    for (unsigned int i = 0;i < dimension;++i)
        lowerBounds[i] = workVec[i];

    workVec = mcmUpdateValue->GetParameterUpperBounds();
    for (unsigned int i = 0;i < dimension;++i)
        upperBounds[i] = workVec[i];

    workVec = mcmUpdateValue->GetParametersAsVector();
    for (unsigned int i = 0;i < dimension;++i)
        p[i] = workVec[i];
    
    costValue = this->PerformSingleOptimization(p,cost,lowerBounds,upperBounds);
    this->GetProfiledInformation(cost,mcmUpdateValue,b0Value,sigmaSqValue);
    
    for (unsigned int i = 0;i < dimension;++i)
        workVec[i] = p[i];

    mcmUpdateValue->SetParametersFromVector(workVec);
    
    mcmValue = mcmUpdateValue;
    aiccValue = this->ComputeAICcValue(mcmValue,costValue);
}

template <class InputPixelType, class OutputPixelType>
void
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>
::SpecificModelEstimation(MCMPointer &mcmValue, std::vector <double> &observedSignals, itk::ThreadIdType threadId,
                          double &aiccValue, double &b0Value, double &sigmaSqValue)
{
    // Finally, we're done with ball and zeppelin, an example of what's next up with multi-tensor
    // - First create model
    MCMCreatorType *mcmCreator = m_MCMCreators[threadId];
    // Other params are supposed to be already initialized from trunk estimation
    mcmCreator->SetCompartmentType(Tensor);

    MCMPointer mcmUpdateValue = mcmCreator->GetNewMultiCompartmentModel();
    MCMPointer mcmOptimalModel = mcmCreator->GetNewMultiCompartmentModel();

    CostFunctionBasePointer cost = this->CreateCostFunction(observedSignals,mcmUpdateValue);

    // - Update multi-tensor model against observed signals
    unsigned int dimension = mcmUpdateValue->GetNumberOfParameters();
    ParametersType p(dimension);
    MCMType::ListType workVec(dimension);
    itk::Array<double> lowerBounds(dimension), upperBounds(dimension);

    workVec = mcmUpdateValue->GetParameterLowerBounds();
    for (unsigned int i = 0;i < dimension;++i)
        lowerBounds[i] = workVec[i];

    workVec = mcmUpdateValue->GetParameterUpperBounds();
    for (unsigned int i = 0;i < dimension;++i)
        upperBounds[i] = workVec[i];

    double optimalCostValue = 0;
    double optimalB0Value = 0;
    double optimalSigmaSqValue = 0;

    unsigned int numIsoCompartments = mcmUpdateValue->GetNumberOfIsotropicCompartments();
    unsigned int numCompartments = mcmUpdateValue->GetNumberOfCompartments();

    unsigned int numNonIsoCompartments = numCompartments - numIsoCompartments;
    SequenceGeneratorType ldsAngles(3 * numNonIsoCompartments);
    MCMType::ListType lowerBoundsAngleSampling(3 * numNonIsoCompartments,0.0);
    MCMType::ListType upperBoundsAngleSampling(3 * numNonIsoCompartments,2.0 * M_PI);
    
    for (unsigned int i = 0;i < numNonIsoCompartments;++i)
    {
        upperBoundsAngleSampling[numNonIsoCompartments + i] = 1.0;
        upperBoundsAngleSampling[2 * numNonIsoCompartments + i] = 1.0;
    }
    
    ldsAngles.SetLowerBounds(lowerBoundsAngleSampling);
    ldsAngles.SetUpperBounds(upperBoundsAngleSampling);
    std::vector <double> sampledData;
    
    for (unsigned int restartNum = 0;restartNum < 3 * m_NumberOfRandomRestarts;++restartNum)
    {
        // - Now the tricky part: initialize from previous model, handled somewhere else
        this->InitializeModelFromSimplifiedOne(mcmValue,mcmUpdateValue);
        
        sampledData = ldsAngles.GetNextSequenceValue();
        
        for (unsigned int i = numIsoCompartments;i < numCompartments;++i)
        {
            unsigned int index = i - numIsoCompartments;
            // Random alpha initialization
            mcmUpdateValue->GetCompartment(i)->SetPerpendicularAngle(sampledData[index]);
            
            // Takes care of initializing d2=l2-l3 (which is 0 in Zeppelin) to a positive realistic value
            double zeppelinAxDiff = mcmUpdateValue->GetCompartment(i)->GetAxialDiffusivity();
            double zeppelinRadDiff = mcmUpdateValue->GetCompartment(i)->GetRadialDiffusivity1();
            
            double w1 = sampledData[numNonIsoCompartments + index];
            mcmUpdateValue->GetCompartment(i)->SetRadialDiffusivity1(w1 * zeppelinRadDiff + (1 - w1) *  zeppelinAxDiff);
            
            double w2 = sampledData[2 * numNonIsoCompartments + index];
            mcmUpdateValue->GetCompartment(i)->SetRadialDiffusivity2(w2 * zeppelinRadDiff + (1 - w2) *  1e-5);
        }

        workVec = mcmUpdateValue->GetParametersAsVector();
        for (unsigned int i = 0;i < dimension;++i)
            p[i] = workVec[i];

        double costValue = this->PerformSingleOptimization(p,cost,lowerBounds,upperBounds);

        if ((costValue < optimalCostValue)||(restartNum == 0))
        {
            // - Get estimated data
            for (unsigned int i = 0;i < dimension;++i)
                workVec[i] = p[i];

            mcmOptimalModel->SetParametersFromVector(workVec);

            this->GetProfiledInformation(cost,mcmOptimalModel,optimalB0Value,optimalSigmaSqValue);
            optimalCostValue = costValue;
        }
    }

    aiccValue = this->ComputeAICcValue(mcmOptimalModel,optimalCostValue);
    mcmValue = mcmOptimalModel;
    b0Value = optimalB0Value;
    sigmaSqValue = optimalSigmaSqValue;
}

template <class InputPixelType, class OutputPixelType>
typename MCMEstimatorImageFilter<InputPixelType, OutputPixelType>::OptimizerPointer
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>
::CreateOptimizer(CostFunctionBasePointer &cost, itk::Array<double> &lowerBounds, itk::Array<double> &upperBounds)
{
    OptimizerPointer returnOpt;
    double xTol = m_XTolerance;
    bool defaultTol = false;
    if (m_XTolerance == 0)
    {
        defaultTol = true;
        xTol = 1.0e-4;
    }


    double maxEvals = m_MaxEval;
    if (m_MaxEval == 0)
        maxEvals = 400 * lowerBounds.GetSize();

    if (m_Optimizer != "levenberg")
    {
        anima::NLOPTOptimizers::Pointer tmpOpt = anima::NLOPTOptimizers::New();

        if (m_MLEstimationStrategy != VariableProjection)
        {
            anima::BaseMCMCostFunction *costCast = dynamic_cast <anima::BaseMCMCostFunction *> (cost.GetPointer());
            tmpOpt->SetCostFunction(costCast);
            tmpOpt->SetAlgorithm(NLOPT_AUGLAG);

            if (m_Optimizer == "bobyqa")
            {
                tmpOpt->SetLocalOptimizer(NLOPT_LN_BOBYQA);
                if (defaultTol)
                    xTol = 1.0e-7;
            }
            else if (m_Optimizer == "ccsaq")
                tmpOpt->SetLocalOptimizer(NLOPT_LD_CCSAQ);

            if (!m_UseFixedWeights)
            {
                typedef anima::MCMWeightsInequalityConstraintFunction WeightInequalityFunctionType;
                WeightInequalityFunctionType::Pointer weightsInequality = WeightInequalityFunctionType::New();
                double wIneqTol = std::min(1.0e-16, xTol / 10.0);
                weightsInequality->SetTolerance(wIneqTol);
                weightsInequality->SetMCMStructure(costCast->GetMCMStructure());

                tmpOpt->AddInequalityConstraint(weightsInequality);
            }
        }
        else
        {
            if (m_Optimizer == "bobyqa")
            {
                tmpOpt->SetAlgorithm(NLOPT_LN_BOBYQA);
                if (defaultTol)
                    xTol = 1.0e-7;
            }
            else if (m_Optimizer == "ccsaq")
                tmpOpt->SetAlgorithm(NLOPT_LD_CCSAQ);

            anima::GaussianMCMVariableProjectionSingleValuedCostFunction *costCast =
                    dynamic_cast <anima::GaussianMCMVariableProjectionSingleValuedCostFunction *> (cost.GetPointer());
            tmpOpt->SetCostFunction(costCast);
        }

        tmpOpt->SetMaximize(false);
        tmpOpt->SetXTolRel(xTol);
        tmpOpt->SetFTolRel(xTol * 1.0e-2);
        tmpOpt->SetMaxEval(maxEvals);
        tmpOpt->SetVectorStorageSize(2000);
        
        if (!m_UseBoundedOptimization)
        {
            tmpOpt->SetLowerBoundParameters(lowerBounds);
            tmpOpt->SetUpperBoundParameters(upperBounds);
        }

        returnOpt = tmpOpt;
    }
    else
    {
        if (m_MLEstimationStrategy != VariableProjection)
            itkExceptionMacro("Levenberg Marquardt optimizer only supported with variable projection");

        typedef itk::LevenbergMarquardtOptimizer LevenbergMarquardtOptimizerType;
        LevenbergMarquardtOptimizerType::Pointer tmpOpt = LevenbergMarquardtOptimizerType::New();

        anima::GaussianMCMVariableProjectionMultipleValuedCostFunction *costCast =
                dynamic_cast <anima::GaussianMCMVariableProjectionMultipleValuedCostFunction *> (cost.GetPointer());
        
        double gTol = m_GTolerance;
        if (m_GTolerance == 0)
            gTol = 1.0e-5;
        
        tmpOpt->SetCostFunction(costCast);
        tmpOpt->SetEpsilonFunction(xTol * 1.0e-3);
        tmpOpt->SetGradientTolerance(gTol);
        tmpOpt->SetNumberOfIterations(maxEvals);
        tmpOpt->SetValueTolerance(xTol);
        tmpOpt->SetUseCostFunctionGradient(!m_VNLDerivativeComputation);
        
        returnOpt = tmpOpt;
    }

    return returnOpt;
}

template <class InputPixelType, class OutputPixelType>
double
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>
::PerformSingleOptimization(ParametersType &p, CostFunctionBasePointer &cost, itk::Array<double> &lowerBounds, itk::Array<double> &upperBounds)
{
    double costValue = this->GetCostValue(cost,p);
    
    OptimizerPointer optimizer = this->CreateOptimizer(cost,lowerBounds,upperBounds);
    
    optimizer->SetInitialPosition(p);
    optimizer->StartOptimization();
        
    p = optimizer->GetCurrentPosition();

    if (!m_UseBoundedOptimization)
    {
        // Takes care of round-off errors in NLOpt resulting
        // in parameters sometimes slightly off bounds
        for (unsigned int i = 0;i < p.GetSize();++i)
        {
            if (p[i] < lowerBounds[i])
                p[i] = lowerBounds[i];

            if (p[i] > upperBounds[i])
                p[i] = upperBounds[i];
        }
    }
        
    costValue = this->GetCostValue(cost,p);
    
    return costValue;
}

template <class InputPixelType, class OutputPixelType>
double
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>
::GetCostValue(CostFunctionBasePointer &cost, ParametersType &p)
{
    if (m_MLEstimationStrategy == VariableProjection)
    {
        if (m_Optimizer == "levenberg")
        {
            anima::GaussianMCMVariableProjectionMultipleValuedCostFunction *costCast =
                    dynamic_cast <anima::GaussianMCMVariableProjectionMultipleValuedCostFunction *> (cost.GetPointer());

            costCast->GetInternalCost()->GetValues(p);
            return costCast->GetInternalCost()->GetCurrentCostValue();
        }

        anima::GaussianMCMVariableProjectionSingleValuedCostFunction *costCast =
                dynamic_cast <anima::GaussianMCMVariableProjectionSingleValuedCostFunction *> (cost.GetPointer());

        return costCast->GetValue(p);
    }

    anima::BaseMCMCostFunction *costCast = dynamic_cast <anima::BaseMCMCostFunction *> (cost.GetPointer());
    return costCast->GetValue(p);
}

template <class InputPixelType, class OutputPixelType>
void
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>
::GetProfiledInformation(CostFunctionBasePointer &cost, MCMPointer &mcm, double &b0Value, double &sigmaSqValue)
{
    if (m_MLEstimationStrategy == VariableProjection)
    {
        if (m_Optimizer == "levenberg")
        {
            anima::GaussianMCMVariableProjectionMultipleValuedCostFunction *costCast =
                    dynamic_cast <anima::GaussianMCMVariableProjectionMultipleValuedCostFunction *> (cost.GetPointer());

            b0Value = costCast->GetB0Value();
            sigmaSqValue = costCast->GetSigmaSquare();
            mcm->SetCompartmentWeights(costCast->GetOptimalWeights());

            return;
        }

        anima::GaussianMCMVariableProjectionSingleValuedCostFunction *costCast =
                dynamic_cast <anima::GaussianMCMVariableProjectionSingleValuedCostFunction *> (cost.GetPointer());

        b0Value = costCast->GetB0Value();
        sigmaSqValue = costCast->GetSigmaSquare();
        mcm->SetCompartmentWeights(costCast->GetOptimalWeights());

        return;
    }

    anima::BaseMCMCostFunction *costCast = dynamic_cast <anima::BaseMCMCostFunction *> (cost.GetPointer());
    b0Value = costCast->GetB0Value();
    sigmaSqValue = costCast->GetSigmaSquare();
}

template <class InputPixelType, class OutputPixelType>
double
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>
::ComputeAICcValue(MCMPointer &mcmValue, double costValue)
{
    // Compute AICu (improvement over AICc)
    // nbparams is the number of parameters of the model plus the B0 value and the variance value
    double nbparams = mcmValue->GetNumberOfParameters() + 2.0;

    // We assume the cost value is returned as - 2 * log-likelihood
    double AICc = costValue + 2.0 * nbparams + 2.0 * nbparams * (nbparams + 1.0) / (m_NumberOfImages - nbparams - 1.0)
            + m_NumberOfImages * std::log(m_NumberOfImages / (m_NumberOfImages - nbparams - 1.0));

    return AICc;
}

template <class InputPixelType, class OutputPixelType>
void
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>
::InitializeStickModelFromDTI(MCMPointer &dtiModel, MCMPointer &complexModel, SequenceGeneratorType &generator)
{
    BaseCompartmentType *tensorCompartment = dtiModel->GetCompartment(0);

    unsigned int numIsotropicComponents = m_ModelWithFreeWaterComponent + m_ModelWithRestrictedWaterComponent + m_ModelWithStationaryWaterComponent;
    unsigned int numNonIsotropicComponents = complexModel->GetNumberOfCompartments() - numIsotropicComponents;
    if (numNonIsotropicComponents > 1)
        this->SampleStickModelCompartmentsFromDTI(tensorCompartment,complexModel,generator);
    else
    {
        complexModel->GetCompartment(numIsotropicComponents)->SetOrientationTheta(tensorCompartment->GetOrientationTheta());
        complexModel->GetCompartment(numIsotropicComponents)->SetOrientationPhi(tensorCompartment->GetOrientationPhi());
    }
}

template <class InputPixelType, class OutputPixelType>
void
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>
::InitializeModelFromSimplifiedOne(MCMPointer &simplifiedModel, MCMPointer &complexModel)
{
    // copy everything
    unsigned int numCompartments = complexModel->GetNumberOfCompartments();
    if (numCompartments != simplifiedModel->GetNumberOfCompartments())
        itkExceptionMacro("Simplified and complex model should have the same number of components.");

    for (unsigned int i = 0;i < numCompartments;++i)
        complexModel->GetCompartment(i)->CopyFromOther(simplifiedModel->GetCompartment(i));

    // Now we're talking about weights
    complexModel->SetCompartmentWeights(simplifiedModel->GetCompartmentWeights());
}

template <class InputPixelType, class OutputPixelType>
void
MCMEstimatorImageFilter<InputPixelType, OutputPixelType>
::SampleStickModelCompartmentsFromDTI(BaseCompartmentType *tensorCompartment, MCMPointer &complexModel,
                                      SequenceGeneratorType &generator)
{
    unsigned int posFirstNonFree = complexModel->GetNumberOfIsotropicCompartments();
    unsigned int numCompartmentsComplex = complexModel->GetNumberOfCompartments() - posFirstNonFree;

    typedef anima::TensorCompartment TensorCompartmentType;
    typedef itk::Matrix<double,3,3> MatrixType;
    TensorCompartmentType *tensorSimplifiedCompartment = dynamic_cast <TensorCompartmentType *> (tensorCompartment);
    if (!tensorSimplifiedCompartment)
        itkExceptionMacro("Simplified model should be DTI, nothing else handled yet");

    MatrixType diffusionTensor = tensorSimplifiedCompartment->GetDiffusionTensor();
    itk::SymmetricEigenAnalysis <MatrixType, vnl_diag_matrix<double>, MatrixType> eigenComputer(3);
    vnl_diag_matrix <double> eigValsTensor(3,0);
    MatrixType eigVecsTensor;

    eigenComputer.ComputeEigenValuesAndVectors(diffusionTensor, eigValsTensor, eigVecsTensor);
    
    bool zeroTensor = true;
    for (unsigned int i = 0;i < 3;++i)
    {
        if (eigValsTensor[i] != 0)
        {
            zeroTensor = false;
            break;
        }
    }

    // Correction in case eigen values are less than 0 (degenerated tensor)
    if (eigValsTensor[0] <= 0)
    {
        for (unsigned int i = 0;i < 3;++i)
            eigValsTensor[i] = std::abs(eigValsTensor[i]);

        vnl_matrix <double> tmpEigVecs = eigVecsTensor.GetVnlMatrix().as_matrix();
        vnl_matrix <double> tmpTensor(3,3);
        MatrixType correctedTensor;
        anima::RecomposeTensor(eigValsTensor,tmpEigVecs,tmpTensor);
        correctedTensor = tmpTensor;

        eigenComputer.ComputeEigenValuesAndVectors(correctedTensor, eigValsTensor, eigVecsTensor);
    }
    
    double linearCoefficient = 0.0;
    double planarCoefficient = 0.0;
    double sphericalCoefficient = 1.0;
    
    if (!zeroTensor)
    {
        linearCoefficient = (eigValsTensor[2] - eigValsTensor[1]) / eigValsTensor[2];
        planarCoefficient = (eigValsTensor[1] - eigValsTensor[0]) / eigValsTensor[2];
        sphericalCoefficient = eigValsTensor[0] / eigValsTensor[2];
    }

    std::vector <double> sampledValues = generator.GetNextSequenceValue();

    MatrixType dcmDirection;
    typedef vnl_vector_fixed<double,3> GradientType;
    GradientType orient_sph, resOrient;
    itk::SymmetricEigenAnalysis <MatrixType, vnl_diag_matrix<double>, MatrixType> dcmComputer(3);
    vnl_diag_matrix <double> eigValsDcm(3,0);
    MatrixType eigVecsDcm;

    for (unsigned int k = 0;k < numCompartmentsComplex;++k)
    {
        dcmDirection.Fill(0);
        anima::TransformSphericalToCartesianCoordinates(sampledValues[numCompartmentsComplex + k],sampledValues[2 * numCompartmentsComplex + k],1.0,orient_sph);

        for (unsigned int i = 0;i < 3;++i)
        {
            for (unsigned int j = i;j < 3;++j)
            {
                dcmDirection(i,j) += (linearCoefficient + planarCoefficient * sampledValues[k]) * eigVecsTensor(2,i) * eigVecsTensor(2,j);
                dcmDirection(i,j) += planarCoefficient * (1.0 - sampledValues[k]) * eigVecsTensor(1,i) * eigVecsTensor(1,j);
                dcmDirection(i,j) += sphericalCoefficient * orient_sph[i] * orient_sph[j];
            }
        }

        for (unsigned int i = 0;i < 3;++i)
            for (unsigned int j = i+1;j < 3;++j)
                dcmDirection(j,i) = dcmDirection(i,j);

        dcmComputer.ComputeEigenValuesAndVectors(dcmDirection, eigValsDcm, eigVecsDcm);

        for (unsigned int i = 0;i < 3;++i)
            resOrient[i] = eigVecsDcm(2,i);

        anima::TransformCartesianToSphericalCoordinates(resOrient, orient_sph);

        complexModel->GetCompartment(posFirstNonFree+k)->SetOrientationTheta(orient_sph[0]);
        complexModel->GetCompartment(posFirstNonFree+k)->SetOrientationPhi(orient_sph[1]);
    }
}

} // end namespace anima
