// My Libraries
#include "Neural Networks.h"

int main()
{
    // Variables
    bool debugConsole = true;
    std::string datasetPath = "\\Image Dataset 0000";
    std::string datasetSession = "Session 0001";
    std::string fileStemArchive = "Archive 0000";
    std::vector<unsigned> networkTopologyHiddenLayers;

    // Load the default training session
    ImageTrainingDataset dataset(datasetPath);
    //ImageTrainingDataset dataset(debugConsole, datasetPath); 

    //// Create a new training session
    //datasetSession = "Session 0001";
    //dataset.createNewTrainingSession(datasetSession);

    //// Load an existing training session
    //ImageTrainingDataset dataset(datasetPath, datasetSession);
    //ImageTrainingDataset dataset(debugConsole, datasetPath, datasetSession);

    // Prepare Training Session
    dataset.prepareTrainingSession();

    // Create New Neural Network
    networkTopologyHiddenLayers.clear();
    networkTopologyHiddenLayers.push_back(4);
    dataset.createNewNeuralNetwork(networkTopologyHiddenLayers);

    // Archive Neural Network
    dataset.archiveNeuralNetwork(fileStemArchive);

    // Exit Program
    return 0;
}