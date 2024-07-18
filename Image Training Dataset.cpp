// Custom Libraries
#include "Neural Networks.h"

// Functions
void exampleImageTrainingDataset(void);

// Program Entry Point
int main()
{
    // Example Using Image Training Dataset Class
    exampleImageTrainingDataset();

    // Exit Program
    return 0;
}

void exampleImageTrainingDataset(void)
{
    // Local Variables Used For The Image Training Dataset
    bool debugConsole = true;
    std::string datasetPath = "\\Image Dataset 0000";
    std::string trainingSession = "Session 0000";
    std::string fileStemArchive = "Archive 0000";
    double neuralNetworkEta = 0.15;
    double neuralNetworkAlpha = 0.5;
    std::vector<unsigned> neuralNetworkHiddenLayersTopology = { 8, 4, 2 };
    double percentTrainingData = 0.75;
    double percentValidationData = 0.20;
    double percentTestData = 0.05;

    // Load The Default Training Session
    ImageTrainingDataset dataset(datasetPath);
    //ImageTrainingDataset dataset(debugConsole, datasetPath); 

    //// Load An Existing Training Session
    //ImageTrainingDataset dataset(datasetPath, trainingSession);
    //ImageTrainingDataset dataset(debugConsole, datasetPath, trainingSession);

    //// Create A New Training Session
    //dataset.createNewTrainingSession(trainingSession);

    //// Update The Training Session
    //dataset.updateTrainingSession(percentTrainingData, percentValidationData, percentTestData);

    //// Prepare The Training Session
    //dataset.prepareTrainingSession(neuralNetworkEta, neuralNetworkAlpha, neuralNetworkHiddenLayersTopology);

    //// Load An Existing Neural Network
    //dataset.loadExistingNeuralNetwork(fileStemArchive);

    //// Create A New Neural Network
    //dataset.createNewNeuralNetwork();

    //// Archive The Neural Network
    //dataset.archiveNeuralNetwork(fileStemArchive);
}
