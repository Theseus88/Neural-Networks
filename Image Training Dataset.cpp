// My Libraries
#include "Neural Networks.h"

int main()
{
    // Local Variables Used For The Image Training Dataset
    bool debugConsole = true;
    std::string datasetPath = "\\Image Dataset 0000";
    std::string trainingSession = "Session 0001";
    std::string fileStemArchive = "Archive 0000";
    double percentTrainingData = 0.75;
    double percentValidationData = 0.20;
    double percentTestData = 0.05;
    std::vector<unsigned> networkTopologyHiddenLayers = {8, 4, 2};
    
    // Load the default training session
    ImageTrainingDataset dataset(datasetPath);
    //ImageTrainingDataset dataset(debugConsole, datasetPath);

    //// Load an existing training session
    //ImageTrainingDataset dataset(datasetPath, trainingSession);
    //ImageTrainingDataset dataset(debugConsole, datasetPath, trainingSession); 

    //// Create a new training session
    //dataset.createNewTrainingSession(trainingSession);

    // Prepare Training Session
    dataset.prepareTrainingSession(percentTrainingData, percentValidationData, percentTestData);

    // Create New Neural Network
    dataset.createNewNeuralNetwork(networkTopologyHiddenLayers);

    // Archive Neural Network
    dataset.archiveNeuralNetwork(fileStemArchive);

    // Exit Program
    return 0;
}
