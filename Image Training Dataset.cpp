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
    std::string datasetPath = "C:\\Users\\Theseus88\\OneDrive\\Desktop\\Image Dataset 0000";
    std::string trainingSession = "Session 0000";
    std::string fileStemArchive = "Archive 0000";
    double neuralNetworkEta = 0.15; // Learning Rate ( 0 < value <= 1 )
    double neuralNetworkAlpha = 0.5; // Momentum ( 0 < value <= 1 )
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
    
    //// Switch To An Existing Training Session 
    //dataset.existingTrainingSession(trainingSession);

    //// Create A New Training Session
    //dataset.newTrainingSession(trainingSession);

    // Update The Training Session (Make sure there are at least three images in the main images directory of the dataset before calling)
    dataset.updateTrainingSession(percentTrainingData, percentValidationData, percentTestData);

    //// Load An Existing Neural Network
    //dataset.existingNeuralNetwork(fileStemArchive);

    // Create A New Neural Network
    dataset.newNeuralNetwork(neuralNetworkEta, neuralNetworkAlpha, neuralNetworkHiddenLayersTopology);

    // For Each Input Image In Training Session Training Images List, Feed Forward And Back Propagate
    dataset.runTrainingImages();

    //// These Two Functions Are Still In Development...
    //dataset.runValidationImages(); // Currently An Empty Function... Will Come Back To Later...
    //dataset.runTestImages(); // Currently An Empty Function... Will Come Back To Later...

    // Archive The Neural Network
    dataset.archiveNeuralNetwork(fileStemArchive);
}
