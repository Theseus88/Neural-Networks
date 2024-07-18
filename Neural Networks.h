#pragma once

// C++ Libraries
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>

// OpenCV Libraries
#include <opencv2/opencv.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>

// Add Comment Here Later
class Neuron;
typedef std::vector<Neuron> Layer;
class NeuralNetwork;
class ImageTrainingDataset;

// Neuron Class
class Neuron
{
public:
	// Constructors
	Neuron(const bool& console, const unsigned& index, const unsigned& outputs, const double& eta, const double& alpha);
	Neuron(const bool& console, const unsigned& index, const unsigned& outputs, const double& eta, const double& alpha, const double& output, const double& gradient);

	// Getters & Setters
	unsigned getIndex(void) const { return _Index; };
	void setIndex(const unsigned& value) { _Index = value; };
	double getEta(void) const { return _Eta; };
	void setEta(const double& value) { _Eta = value; };
	double getAlpha(void) const { return _Alpha; };
	void setAlpha(const double& value) { _Alpha = value; };
	double getOutput(void) const { return _Output; };
	void setOutput(const double& value) { _Output = value; };
	double getGradient(void) const { return _Gradient; };
	void setGradient(const double& value) { _Gradient = value; };
	double getConnectionWeight(const unsigned& index) { return _OutputWeights[index].weight; };
	void setConnectionWeight(const unsigned& index, const double& value) { _OutputWeights[index].weight = value; };
	double getConnectionDeltaWeight(const unsigned& index) { return _OutputWeights[index].deltaWeight; };
	void setConnectionDeltaWeight(const unsigned& index, const double& value) { _OutputWeights[index].deltaWeight = value; };

	// Forward propagation
	void feedForward(const Layer& previousLayer);

	// Back propagation
	void calculateOutputGradients(double targetValue);
	void calculateHiddenGradients(const Layer& nextLayer);
	void updateInputWeights(Layer& previousLayer);

private:
	// Variables
	struct Connection {
		double weight;
		double deltaWeight;
	};
	unsigned _Index;
	double _Eta;
	double _Alpha;
	double _Output;
	double _Gradient;
	std::vector<Connection> _OutputWeights;

	// Functions
	static double transferFunction(double value);
	static double transferFunctionDerivative(double value);
	double sumDOW(const Layer& nextLayer) const;

};
Neuron::Neuron(const bool& console, const unsigned& index, const unsigned& outputs, const double& eta, const double& alpha)
{
	if (console) {
		std::cout << "\tNeuron " << index << " added to the network layer.\n";
	}
	setIndex(index);
	for (unsigned output = 0; output < outputs; output++) {
		_OutputWeights.push_back(Connection());
		_OutputWeights.back().weight = rand() / double(RAND_MAX);
		_OutputWeights.back().deltaWeight = 1.0;
		if (console) {
			std::cout << "\t\tConnection " << output << " added to the neuron.\n";
		}
	}
	setEta(eta);
	setAlpha(alpha);
	setOutput(0.0);
	setGradient(0.0);
}
Neuron::Neuron(const bool& console, const unsigned& index, const unsigned& outputs, const double& eta, const double& alpha, const double& output, const double& gradient)
{
	if (console) {
		std::cout << "\tNeuron " << index << " added to the network layer.\n";
	}
	setIndex(index);
	for (unsigned output = 0; output < outputs; output++) {
		_OutputWeights.push_back(Connection());
		_OutputWeights.back().weight = rand() / double(RAND_MAX);
		_OutputWeights.back().deltaWeight = 1.0;
		if (console) {
			std::cout << "\t\tConnection " << output << " added to the neuron.\n";
		}
	}
	setEta(eta);
	setAlpha(alpha);
	setOutput(output);
	setGradient(gradient);
}
void Neuron::feedForward(const Layer& previousLayer)
{
	double sum = 0.0;
	for (unsigned neuronIndex = 0; neuronIndex < previousLayer.size(); neuronIndex++) {
		sum += previousLayer[neuronIndex].getOutput() * previousLayer[neuronIndex]._OutputWeights[_Index].weight;
	}
	_Output = Neuron::transferFunction(sum);
}
double Neuron::transferFunction(double value)
{
	return tanh(value);
}
double Neuron::transferFunctionDerivative(double value)
{
	return 1.0 - tanh(value) * tanh(value) * value;
}
void Neuron::calculateOutputGradients(double targetValue)
{
	double delta = targetValue - _Output;
	_Gradient = delta * Neuron::transferFunctionDerivative(_Output);
}
void Neuron::calculateHiddenGradients(const Layer& nextLayer)
{
	double dow = sumDOW(nextLayer);
	_Gradient = dow * Neuron::transferFunctionDerivative(_Output);
}
double Neuron::sumDOW(const Layer& nextLayer) const
{
	double sum = 0.0;
	for (unsigned neuronIndex = 0; neuronIndex < nextLayer.size() - 1; neuronIndex++) {
		sum += _OutputWeights[neuronIndex].weight * nextLayer[neuronIndex]._Gradient;
	}
	return sum;
}
void Neuron::updateInputWeights(Layer& previousLayer)
{
	for (unsigned neuronIndex = 0; neuronIndex < previousLayer.size(); neuronIndex++) {
		Neuron& neuron = previousLayer[neuronIndex];
		double oldDeltaWeight = neuron._OutputWeights[_Index].deltaWeight;
		double newDeltaWeight = _Eta * neuron.getOutput() * _Gradient + _Alpha * oldDeltaWeight;
		neuron._OutputWeights[_Index].deltaWeight = newDeltaWeight;
		neuron._OutputWeights[_Index].weight += newDeltaWeight;
	}
}

// Neural Network Class
class NeuralNetwork
{
	// Private Variables
	bool debugConsole;
	std::vector<Layer> networkLayers;
	double networkError;
	double networkRecentAverageError;
	double networkRecentAverageSmoothingFactor;

public:
	// Constructors
	NeuralNetwork(void) { 
		debugConsole = true;
		networkError = 0.0;
		networkRecentAverageError = 0.0;
		networkRecentAverageSmoothingFactor = 100.0;
		networkLayers.clear();
	};

	// Public Functions
	void feedForward(const std::vector<double>& inputValues);
	void backPropagation(const std::vector<double>& targetValues);
	void getResults(std::vector<double>& resultValues) const;
	double getRecentAverageError(void) const { return networkRecentAverageError; }

	// Working On
	void createNewNeuralNetwork(const std::vector<unsigned>& topology, const double& eta, const double& alpha);
	void loadExistingNeuralNetwork(std::vector<unsigned>& topology, const std::string& filePathArchive);
	void archiveNeuralNetwork(const std::vector<unsigned>& topology, const std::string& filePathArchive);

};
void NeuralNetwork::feedForward(const std::vector<double>& inputValues)
{
	assert(inputValues.size() == networkLayers[0].size() - 1);

	// Assign (latch) the input values into the input neurons
	for (unsigned i = 0; i < inputValues.size(); i++) {
		networkLayers[0][i].setOutput(inputValues[i]);
	}

	// Forward propagation
	for (unsigned layerIndex = 1; layerIndex < networkLayers.size(); layerIndex++) {
		Layer& previousLayer = networkLayers[layerIndex - 1];
		for (unsigned neuronIndex = 0; neuronIndex < networkLayers[layerIndex].size() - 1; neuronIndex++) {
			networkLayers[layerIndex][neuronIndex].feedForward(previousLayer);
		}
	}
}
void NeuralNetwork::backPropagation(const std::vector<double>& targetValues)
{
	// Calculate overall net error (RMS of output neuron errors)
	Layer& outputLayer = networkLayers.back();
	networkError = 0.0;
	for (unsigned neuronIndex = 0; neuronIndex < outputLayer.size() - 1; neuronIndex++) {
		double delta = targetValues[neuronIndex] - outputLayer[neuronIndex].getOutput();
		networkError += delta * delta;
	}
	networkError /= outputLayer.size() - 1; // Get average error squared
	networkError = sqrt(networkError); // RMS

	// Implement a recent average measurement (Has nothing to do with the actual neural network)
	networkRecentAverageError = (networkRecentAverageError * networkRecentAverageSmoothingFactor + networkError)
		/ (networkRecentAverageSmoothingFactor + 1.0);

	// Calculate output layer gradients
	for (unsigned neuronIndex = 0; neuronIndex < outputLayer.size() - 1; neuronIndex++) {
		outputLayer[neuronIndex].calculateOutputGradients(targetValues[neuronIndex]);
	}

	// Calculate gradients on hidden layers
	for (unsigned layerNumber = networkLayers.size() - 2; layerNumber > 0; layerNumber--) {
		Layer& hiddenLayer = networkLayers[layerNumber];
		Layer& nextLayer = networkLayers[layerNumber + 1];
		for (unsigned neuronIndex = 0; neuronIndex < hiddenLayer.size(); neuronIndex++) {
			hiddenLayer[neuronIndex].calculateHiddenGradients(nextLayer);
		}
	}

	// For all layers from outputs to first hidden layer update connection weights
	for (unsigned layerNumber = networkLayers.size() - 1; layerNumber > 0; layerNumber--) {
		Layer& layer = networkLayers[layerNumber];
		Layer& previousLayer = networkLayers[layerNumber - 1];
		for (unsigned neuronIndex = 0; neuronIndex < layer.size() - 1; neuronIndex++) {
			layer[neuronIndex].updateInputWeights(previousLayer);
		}
	}
}
void NeuralNetwork::getResults(std::vector<double>& resultValues) const
{
	resultValues.clear();
	for (unsigned neuronIndex = 0; neuronIndex < networkLayers.back().size() - 1; neuronIndex++) {
		resultValues.push_back(networkLayers.back()[neuronIndex].getOutput());
	}
}
void NeuralNetwork::createNewNeuralNetwork(const std::vector<unsigned>& topology, const double& eta, const double& alpha)
{
	if (debugConsole) { std::cout << "Update NN.000: Creating a new neural network for the training dataset.\n"; }

	// Add Comment Later
	networkLayers.clear();
	networkError = 0.0;
	networkRecentAverageError = 0.0;
	networkRecentAverageSmoothingFactor = 100.0;

	// Add layers to the neural network.
	for (unsigned layerIndex = 0; layerIndex < topology.size(); layerIndex++) {
		networkLayers.push_back(Layer());
		if (debugConsole) {
			std::cout << "Update NN.000: Layer " << layerIndex << " added to the neural network.\n";
		}

		// Get the number of neurons in the next layer of the neural network.
		unsigned numberOfOutputs = layerIndex == topology.size() - 1 ? 0 : topology[layerIndex + 1];

		// Add neurons to the current layer of the neural network.
		for (unsigned neuronIndex = 0; neuronIndex <= topology[layerIndex]; neuronIndex++) {
			networkLayers.back().push_back(Neuron(debugConsole, neuronIndex, numberOfOutputs, eta, alpha));
		}

		// Force the current layer's bias node's output value to 1.0. It is the last neuron created above.
		networkLayers.back().back().setOutput(1.0);
	}

	if (debugConsole) { std::cout << "Update NN.000: Finished creating the new neural network for the training dataset.\n"; }
}
void NeuralNetwork::loadExistingNeuralNetwork(std::vector<unsigned>& topology, const std::string& filePathArchive)
{
	if (debugConsole) { std::cout << "Update NN.000: Loading an existing neural network for the training dataset.\n"; }
	
	// Add Comment Later
	std::fstream fileArchive;
	topology.clear();
	networkLayers.clear();
	networkError = 0.0;
	networkRecentAverageError = 0.0;
	networkRecentAverageSmoothingFactor = 100.0;

	// Add Comment Later
	if (filePathArchive.empty()) {
		if (debugConsole) { std::cout << "Error NN.000: An empty path was provided for the neural network archive.\n"; }
		abort();
	}
	if (!std::filesystem::exists(filePathArchive)) {
		if (debugConsole) { std::cout << "Error NN.000: \n"; }
		abort();
	}
	fileArchive.open(filePathArchive, std::ios::in);
	if (fileArchive.fail()) {
		if (debugConsole) { std::cout << "Error NN.000: \n"; }
		abort();
	}

	// Get Neural Network Topology
	std::string line;
	int index;
	std::string label;
	std::string valueString;
	unsigned valueUnsigned;
	std::getline(fileArchive, line);
	if (line.empty()) {
		if (debugConsole) { std::cout << "Error NN.000: Neural network archive has no content.\n"; }
		abort();
	}
	index = line.find(" ");
	if (index != 9) {
		if (debugConsole) { std::cout << "Error NN.000: Topology setting is missing.\n"; }
		abort();
	}
	label = line.substr(0, index);
	if (label.compare("Topology:") != 0 && label.compare("topology:") != 0) {
		if (debugConsole) { std::cout << "Error NN.000: Topology setting is missing.\n"; }
		abort();
	}
	line.erase(0, 10);
	if (line.empty()) {
		if (debugConsole) { std::cout << "Error NN.000: Topology setting is empty.\n"; }
		abort();
	}
	while (index != -1) {
		index = line.find(",");
		valueString = line.substr(0, index);
		line.erase(0, index + 1);
		valueUnsigned = std::stoi(valueString) - 1;
		topology.push_back(valueUnsigned);
	}

	// Add Comment Later
	double eta;
	double alpha;
	double output;
	double gradient;
	double weight;
	double deltaWeight;

	// Add layers to the neural network.
	for (unsigned layerIndex = 0; layerIndex < topology.size(); layerIndex++) {
		networkLayers.push_back(Layer());
		if (debugConsole) { std::cout << "Update NN.000: Layer " << layerIndex << " added to the neural network.\n"; }

		// Get the number of neurons in the next layer of the neural network.
		unsigned numberOfOutputs = layerIndex == topology.size() - 1 ? 0 : topology[layerIndex + 1];

		// Add neurons to the current layer of the neural network.
		for (unsigned neuronIndex = 0; neuronIndex <= topology[layerIndex]; neuronIndex++) {

			// Get neuron eta and alpha.
			std::getline(fileArchive, line);
			if (line.empty()) {
				if (debugConsole) { std::cout << "Error NN.000: The neural network archive is missing content.\n"; }
				abort();
			}
			for (int i = 0; i < 2; i++) {
				index = line.find(",");
				line.erase(0, index + 1);
			}
			index = line.find(",");
			eta = std::stod(line.substr(0, index));
			line.erase(0, index + 1);
			index = line.find(",");
			alpha = std::stod(line.substr(0, index));
			line.erase(0, index + 1);
			index = line.find(",");
			output = std::stod(line.substr(0, index));
			line.erase(0, index + 1);
			index = line.find(",");
			gradient = std::stod(line.substr(0, index));
			line.erase(0, index + 1);

			// Create neuron.
			networkLayers.back().push_back(Neuron(debugConsole, neuronIndex, numberOfOutputs, eta, alpha, output, gradient));

			// Update neuron connections' output weights.
			for (int i = 0; i < numberOfOutputs; i++) {
				index = line.find(",");
				weight = std::stod(line.substr(0, index));
				line.erase(0, index + 1);
				index = line.find(",");
				deltaWeight = std::stod(line.substr(0, index));
				line.erase(0, index + 1);
				networkLayers[layerIndex][neuronIndex].setConnectionWeight(i, weight);
				networkLayers[layerIndex][neuronIndex].setConnectionDeltaWeight(i, deltaWeight);
			}
		}
	}
	fileArchive.close();
	if (debugConsole) { std::cout << "Update NN.000: Finished creating the neural network for the training dataset.\n"; }
}
void NeuralNetwork::archiveNeuralNetwork(const std::vector<unsigned>& topology, const std::string& filePathArchive)
{
	if (debugConsole) { std::cout << "Update NN.000: Archiving the neural network.\n"; }

	// Local Variables
	std::fstream fileArchive;
	// Add Comment Later
	if (filePathArchive.empty()) {
		if (debugConsole) { std::cout << "Error NN.000: An empty path was provided for the neural network archive.\n"; }
		abort();
	}
	fileArchive.open(filePathArchive, std::ios::out);
	if (fileArchive.fail()) {
		if (debugConsole) { std::cout << "Error NN.000: Failed to open the neural network archive.\n"; }
		abort();
	}

	// Add Comment Later
	fileArchive << "Topology: ";
	for (unsigned topologyIndex = 0; topologyIndex < topology.size(); topologyIndex++) {
		fileArchive << topology[topologyIndex] + 1 << (topologyIndex != topology.size() - 1 ? "," : "\n");
	}

	// Add Comment Later
	for (unsigned layerIndex = 0; layerIndex < topology.size(); layerIndex++) {

		// Get the number of neurons in the next layer of the neural network.
		unsigned numberOfOutputs = layerIndex == topology.size() - 1 ? 0 : topology[layerIndex + 1];

		// Add Comment Later
		for (unsigned neuronIndex = 0; neuronIndex <= topology[layerIndex]; neuronIndex++) {
			fileArchive << std::to_string(layerIndex) << "," << std::to_string(neuronIndex) << ","
				<< networkLayers[layerIndex][neuronIndex].getEta() << "," << networkLayers[layerIndex][neuronIndex].getAlpha() << ","
				<< networkLayers[layerIndex][neuronIndex].getOutput() << "," << networkLayers[layerIndex][neuronIndex].getGradient();

			// For each neuron output connection get the connection's weights and write to archive file.
			for (unsigned connectionIndex = 0; connectionIndex < numberOfOutputs; connectionIndex++) {
				fileArchive << "," << std::to_string(networkLayers[layerIndex][neuronIndex].getConnectionWeight(connectionIndex)) << ","
					<< std::to_string(networkLayers[layerIndex][neuronIndex].getConnectionDeltaWeight(connectionIndex));
			}
			fileArchive << std::endl;
		}
	}
	fileArchive.close();
	if (debugConsole) {
		std::cout << "Update NN.000: Finished archiving the neural network.\n";
	}
}

// Image Training Dataset Class
class ImageTrainingDataset
{
	// Private Variables - Image Training Dataset
	bool debugConsole;
	std::string directoryImageTrainingDataset;
	std::string directoryImages;
	std::string directoryImageAnnotations;
	std::string directoryTrainingSessions;
	std::string directoryCurrentTrainingSession;
	std::string directoryTrainingSessionImages;
	std::string directoryTrainingSessionImageAnnotations;
	std::string directoryTrainingSessionTargetOutputDataFiles;
	std::string directoryTrainingSessionNeuralNetworkArchives;
	std::string filePathTrainingSessionAnnotationList;
	std::string filePathTrainingSessionTrainingImagesList;
	std::string filePathTrainingSessionValidationImagesList;
	std::string filePathTrainingSessionTestImagesList;
	std::string filePathTrainingSessionDatasetConfiguration;
	std::vector<std::string> trainingSessionAnnotationList;
	unsigned trainingSessionImageWidth;
	unsigned trainingSessionImageHeight;

	// Private Variables - Neural Network
	NeuralNetwork trainingSessionNeuralNetwork;
	double trainingSessionNeuralNetworkEta;
	double trainingSessionNeuralNetworkAlpha;
	std::vector<unsigned> trainingSessionNeuralNetworkTopology;

public:
	// Constructors
	ImageTrainingDataset(const std::string& path) { initializeImageTrainingDataset(true, path); };
	ImageTrainingDataset(const std::string& path, const std::string& session) { initializeImageTrainingDataset(true, path, session); }; // This constructor is not working correctly...
	ImageTrainingDataset(const bool& console, const std::string& path) { initializeImageTrainingDataset(console, path); };
	ImageTrainingDataset(const bool& console, const std::string& path, const std::string& session) { initializeImageTrainingDataset(console, path, session); };

	// Public Functions - Image Training Dataset
	void createNewTrainingSession(const std::string& path);
	void selectExistingTrainingSession(const std::string& path);

	void updateDirectoryTrainingSessionImages(void);
	void updateDirectoryTrainingSessionImageAnnotations(void);
	void updateFileTrainingSessionAnnotationList(void);
	void updateTrainingSessionTargetOutputDataFiles(void);
	void updateTrainingSessionTrainingValidationAndTestFiles(const double& percentTrainingData, const double& percentValidationData, const double& percentTestData);
	void updateTrainingSession(const double& percentTrainingData, const double& percentValidationData, const double& percentTestData);

	void verifyTrainingSessionImageSizesMatch(void);
	void updateFileTrainingSessionDatasetConfiguration(void);	
	void prepareTrainingSession(const double& eta, const double& alpha, const std::vector<unsigned>& hiddenLayersTopology);

	// Public Functions - Neural Network
	void createNewNeuralNetwork(void);
	void loadExistingNeuralNetwork(const std::string& fileStem);
	void archiveNeuralNetwork(const std::string& fileStem);

	// Working On These Three Functions Currently
	void runTrainingImages(void);
	void runValidationImages(void);
	void runTestImages(void);

	// Getters
	bool getDebugConsole(void) const { return debugConsole; };
	std::string getDirectoryImageTrainingDataset(void) const { return directoryImageTrainingDataset; };
	std::string getDirectoryImages(void) const { return directoryImages; };
	std::string getDirectoryImageAnnotations(void) const { return directoryImageAnnotations; };
	std::string getDirectoryTrainingSessions(void) const { return directoryTrainingSessions; };
	std::string getDirectoryCurrentTrainingSession(void) const { return directoryCurrentTrainingSession; };
	std::string getDirectoryTrainingSessionImages(void) const { return directoryTrainingSessionImages; };
	std::string getDirectoryTrainingSessionImageAnnotations(void) const { return directoryTrainingSessionImageAnnotations; };
	std::string getDirectoryTrainingSessionTargetOutputDataFiles(void) const { return directoryTrainingSessionTargetOutputDataFiles; };
	std::string getDirectoryTrainingSessionNeuralNetworkArchives(void) const { return directoryTrainingSessionNeuralNetworkArchives; };
	std::string getFilePathTrainingSessionAnnotationList(void) const { return filePathTrainingSessionAnnotationList; };
	std::string getFilePathTrainingSessionTrainingImagesList(void) const { return filePathTrainingSessionTrainingImagesList; };
	std::string getFilePathTrainingSessionValidationImagesList(void) const { return filePathTrainingSessionValidationImagesList; };
	std::string getFilePathTrainingSessionTestImagesList(void) const { return filePathTrainingSessionTestImagesList; };
	std::string getFilePathTrainingSessionDatasetConfiguration(void) const { return filePathTrainingSessionDatasetConfiguration; };
	std::vector<std::string> getTrainingSessionAnnotationList(void) const { return trainingSessionAnnotationList; };
	unsigned getTrainingSessionImageWidth(void) const { return trainingSessionImageWidth; };
	unsigned getTrainingSessionImageHeight(void) const { return trainingSessionImageHeight; };
	std::vector<unsigned> getTrainingSessionNeuralNetworkTopology(void) const { return trainingSessionNeuralNetworkTopology; };
	double getTrainingSessionNeuralNetworkEta(void) const { return trainingSessionNeuralNetworkEta; };
	double getTrainingSessionNeuralNetworkAlpha(void) const { return trainingSessionNeuralNetworkAlpha; };

	// Setters
	void setDebugConsole(const bool& console);
	void setDirectoryImageTrainingDataset(const std::string& path);
	void setDirectoryImages(const std::string& path);
	void setDirectoryImageAnnotations(const std::string& path);
	void setDirectoryTrainingSessions(const std::string& path);
	void setDirectoryCurrentTrainingSession(const std::string& path);
	void setDirectoryTrainingSessionImages(const std::string& path);
	void setDirectoryTrainingSessionImageAnnotations(const std::string& path);
	void setDirectoryTrainingSessionTargetOutputDataFiles(const std::string& path);
	void setDirectoryTrainingSessionNeuralNetworkArchives(const std::string& path);
	void setFilePathTrainingSessionAnnotationList(const std::string& path);
	void setFilePathTrainingSessionTrainingImagesList(const std::string& path);
	void setFilePathTrainingSessionValidationImagesList(const std::string& path);
	void setFilePathTrainingSessionTestImagesList(const std::string& path);
	void setFilePathTrainingSessionDatasetConfiguration(const std::string& path);
	void setTrainingSessionAnnotationList(void);
	void setTrainingSessionImageWidth(const unsigned& width);
	void setTrainingSessionImageHeight(const unsigned& height);
	void setTrainingSessionNeuralNetworkTopology(const std::vector<unsigned>& hiddenLayersTopology);
	void setTrainingSessionNeuralNetworkEta(const double& eta);
	void setTrainingSessionNeuralNetworkAlpha(const double& alpha);

private:
	// Private Functions
	void initializeImageTrainingDataset(const bool& console, const std::string& path);
	void initializeImageTrainingDataset(const bool& console, const std::string& path, const std::string& session);
	void initializeTrainingSessionNeuralNetwork(void);

};

void ImageTrainingDataset::initializeImageTrainingDataset(const bool& console, const std::string& path)
{
	std::cout << "option 1\n";
	setDebugConsole(console);
	if (debugConsole) { std::cout << "Update ITD.000: Initializing the image training dataset.\n"; }
	setDirectoryImageTrainingDataset(path);
	setDirectoryImages(directoryImageTrainingDataset + "Images\\");
	setDirectoryImageAnnotations(directoryImages + "Image Annotations\\");
	setDirectoryTrainingSessions(directoryImageTrainingDataset + "Training Sessions\\");
	setDirectoryCurrentTrainingSession(directoryTrainingSessions + "Session 0000\\");
	setDirectoryTrainingSessionImages(directoryCurrentTrainingSession + "Images\\");
	setDirectoryTrainingSessionImageAnnotations(directoryTrainingSessionImages + "Image Annotations\\");
	setDirectoryTrainingSessionTargetOutputDataFiles(directoryCurrentTrainingSession + "Target Output Data Files\\");
	setDirectoryTrainingSessionNeuralNetworkArchives(directoryCurrentTrainingSession + "Neural Network Archives\\");
	setFilePathTrainingSessionAnnotationList(directoryCurrentTrainingSession);
	setFilePathTrainingSessionTrainingImagesList(directoryCurrentTrainingSession);
	setFilePathTrainingSessionValidationImagesList(directoryCurrentTrainingSession);
	setFilePathTrainingSessionTestImagesList(directoryCurrentTrainingSession);
	setFilePathTrainingSessionDatasetConfiguration(directoryCurrentTrainingSession);	
	setTrainingSessionAnnotationList();
	verifyTrainingSessionImageSizesMatch();
	initializeTrainingSessionNeuralNetwork();
	updateFileTrainingSessionDatasetConfiguration();
	if (debugConsole) { std::cout << "Update ITD.000: Finished initializing the image training dataset.\n"; }
}
void ImageTrainingDataset::initializeImageTrainingDataset(const bool& console, const std::string& path, const std::string& session)
{
	std::cout << "option 2\n";
	setDebugConsole(console);
	if (debugConsole) { std::cout << "Update ITD.000: Initializing the image training dataset.\n"; }
	setDirectoryImageTrainingDataset(path);
	setDirectoryImages(directoryImageTrainingDataset + "Images\\");
	setDirectoryImageAnnotations(directoryImages + "Image Annotations\\");
	setDirectoryTrainingSessions(directoryImageTrainingDataset + "Training Sessions\\");
	selectExistingTrainingSession(session);
	initializeTrainingSessionNeuralNetwork();
	updateFileTrainingSessionDatasetConfiguration();
	if (debugConsole) { std::cout << "Update ITD.000: Finished initializing the image training dataset.\n"; }
}
void ImageTrainingDataset::initializeTrainingSessionNeuralNetwork(void)
{
	if (debugConsole) { std::cout << "Update ITD.000: \n"; };
	trainingSessionNeuralNetworkTopology.clear();
	std::fstream fileDatasetConfiguration;
	fileDatasetConfiguration.open(filePathTrainingSessionDatasetConfiguration, std::ios::in);
	if (fileDatasetConfiguration.fail()) {
		if (debugConsole) { std::cout << "Error ITD.000: \n"; };
		abort();
	}
	std::string line;
	std::getline(fileDatasetConfiguration, line);
	if (fileDatasetConfiguration.fail() || line.empty()) {
		trainingSessionNeuralNetworkEta = 0.15;
		trainingSessionNeuralNetworkAlpha = 0.5;
	}
	else {
		for (unsigned index = 0; index < 45; index++) {
			std::getline(fileDatasetConfiguration, line);
			if (fileDatasetConfiguration.fail()) {
				if (debugConsole) { std::cout << "Error ITD.000: \n"; }
				abort();
			}
		}
		line.erase(0, 37);
		double value = std::stod(line);
		value == 0 ? trainingSessionNeuralNetworkEta = 0.15 : trainingSessionNeuralNetworkEta = value;
		std::getline(fileDatasetConfiguration, line);
		if (fileDatasetConfiguration.fail() || line.empty()) {
			if (debugConsole) { std::cout << "Error ITD.000: \n"; }
			abort();
		}
		line.erase(0, 39);
		value = std::stod(line);
		value == 0 ? trainingSessionNeuralNetworkAlpha = 0.5 : trainingSessionNeuralNetworkAlpha = value;
	}
	fileDatasetConfiguration.close();
	if (debugConsole) { std::cout << "Update ITD.000: \n"; };
}

void ImageTrainingDataset::setDebugConsole(const bool& console)
{
	debugConsole = console;
}
void ImageTrainingDataset::setDirectoryImageTrainingDataset(const std::string& path)
{
	if (debugConsole) { std::cout << "Update U.003: Setting the image training dataset directory.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error TD.001: An empty path was provided for the image training dataset directory.\n"; }
		abort();
	}
	path.find_last_of("\\") == path.length() - 1 ? directoryImageTrainingDataset = path : directoryImageTrainingDataset = path + "\\";
	if (!std::filesystem::exists(directoryImageTrainingDataset)) {
		if (debugConsole) { std::cout << "Update U.004: Creating the image training dataset directory.\n"; }
		std::filesystem::create_directory(directoryImageTrainingDataset);
		if (debugConsole) { std::cout << "Update U.005: Finished creating the image training dataset directory.\n"; }
	}
	if (debugConsole) { std::cout << "Update U.006: Finished setting the image training dataset directory.\n"; }
}
void ImageTrainingDataset::setDirectoryImages(const std::string& path)
{
	if (debugConsole) { std::cout << "Update U.007: Setting the images directory.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error TD.002: An empty path was provided for the images directory.\n"; }
		abort();
	}
	path.find_last_of("\\") == path.length() - 1 ? directoryImages = path : directoryImages = path + "\\";
	if (!std::filesystem::exists(directoryImages)) {
		if (debugConsole) { std::cout << "Update U.008: Creating the images directory.\n"; }
		std::filesystem::create_directory(directoryImages);
		if (debugConsole) { std::cout << "Update U.009: Finished creating the images directory.\n"; }
	}
	if (debugConsole) { std::cout << "Update U.010: Finished setting the images directory.\n"; }
}
void ImageTrainingDataset::setDirectoryImageAnnotations(const std::string& path)
{
	if (debugConsole) { std::cout << "Update U.011: Setting the image annotations directory.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error TD.003: An empty path was provided for the image annotations directory.\n"; }
		abort();
	}
	path.find_last_of("\\") == path.length() - 1 ? directoryImageAnnotations = path : directoryImageAnnotations = path + "\\";
	if (!std::filesystem::exists(directoryImageAnnotations)) {
		if (debugConsole) { std::cout << "Update U.012: Creating the image annotations directory.\n"; }
		std::filesystem::create_directory(directoryImageAnnotations);
		if (debugConsole) { std::cout << "Update U.013: Finished creating the image annotations directory.\n"; }
	}
	if (debugConsole) { std::cout << "Update U.014: Finished setting the image annotations directory.\n"; }
}
void ImageTrainingDataset::setDirectoryTrainingSessions(const std::string& path)
{
	if (debugConsole) { std::cout << "Update U.000: Setting the training sessions directory.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error TD.000: An empty path was provided for the training sessions directory.\n"; }
		abort();
	}
	path.find_last_of("\\") == path.length() - 1 ? directoryTrainingSessions = path : directoryTrainingSessions = path + "\\";
	if (!std::filesystem::exists(directoryTrainingSessions)) {
		if (debugConsole) { std::cout << "Update U.000: Creating the training sessions directory.\n"; }
		std::filesystem::create_directory(directoryTrainingSessions);
		if (debugConsole) { std::cout << "Update U.000: Finished creating the training sessions directory.\n"; }
	}
	if (debugConsole) { std::cout << "Update U.000: Finished setting the training sessions directory.\n"; }
}
void ImageTrainingDataset::setDirectoryCurrentTrainingSession(const std::string& path)
{
	if (debugConsole) { std::cout << "Update U.000: Setting the current training session directory.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error TD.000: An empty path was provided for the current training session directory.\n"; }
		abort();
	}
	path.find_last_of("\\") == path.length() - 1 ? directoryCurrentTrainingSession = path : directoryCurrentTrainingSession = path + "\\";
	if (!std::filesystem::exists(directoryCurrentTrainingSession)) {
		if (debugConsole) { std::cout << "Update U.000: Creating a new training session directory.\n"; }
		std::filesystem::create_directory(directoryCurrentTrainingSession);
		if (debugConsole) { std::cout << "Update U.000: Finished creating a new training session directory.\n"; }
	}
	if (debugConsole) { std::cout << "Update U.000: Finished setting the current training session directory.\n"; }
}
void ImageTrainingDataset::setDirectoryTrainingSessionImages(const std::string& path)
{
	if (debugConsole) { std::cout << "Update U.007: Setting the training session images directory.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error TD.002: An empty path was provided for the training session images directory.\n"; }
		abort();
	}
	path.find_last_of("\\") == path.length() - 1 ? directoryTrainingSessionImages = path : directoryTrainingSessionImages = path + "\\";
	if (!std::filesystem::exists(directoryTrainingSessionImages)) {
		if (debugConsole) { std::cout << "Update U.008: Creating the training session images directory.\n"; }
		std::filesystem::create_directory(directoryTrainingSessionImages);
		if (debugConsole) { std::cout << "Update U.009: Finished creating the training session images directory.\n"; }
	}
	if (debugConsole) { std::cout << "Update U.010: Finished setting the training session images directory.\n"; }
}
void ImageTrainingDataset::setDirectoryTrainingSessionImageAnnotations(const std::string& path)
{
	if (debugConsole) { std::cout << "Update U.011: Setting the training session image annotations directory.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error TD.003: An empty path was provided for the training session image annotations directory.\n"; }
		abort();
	}
	path.find_last_of("\\") == path.length() - 1 ? directoryTrainingSessionImageAnnotations = path : directoryTrainingSessionImageAnnotations = path + "\\";
	if (!std::filesystem::exists(directoryTrainingSessionImageAnnotations)) {
		if (debugConsole) { std::cout << "Update U.012: Creating the training session image annotations directory.\n"; }
		std::filesystem::create_directory(directoryTrainingSessionImageAnnotations);
		if (debugConsole) { std::cout << "Update U.013: Finished creating the training session image annotations directory.\n"; }
	}
	if (debugConsole) { std::cout << "Update U.014: Finished setting the training session image annotations directory.\n"; }
}
void ImageTrainingDataset::setDirectoryTrainingSessionTargetOutputDataFiles(const std::string& path)
{
	if (debugConsole) { std::cout << "Update U.015: Setting the training session target output data files directory.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error TD.004: An empty path was provided for the training session target output data files directory.\n"; }
		abort();
	}
	path.find_last_of("\\") == path.length() - 1 ? directoryTrainingSessionTargetOutputDataFiles = path : directoryTrainingSessionTargetOutputDataFiles = path + "\\";
	if (!std::filesystem::exists(directoryTrainingSessionTargetOutputDataFiles)) {
		if (debugConsole) { std::cout << "Update U.016: Creating the training session target output data files directory.\n"; }
		std::filesystem::create_directory(directoryTrainingSessionTargetOutputDataFiles);
		if (debugConsole) { std::cout << "Update U.017: Finished creating the training session target output data files directory.\n"; }
	}
	if (debugConsole) { std::cout << "Update U.018: Finished setting the training session target output data files directory.\n"; }
}
void ImageTrainingDataset::setDirectoryTrainingSessionNeuralNetworkArchives(const std::string& path)
{
	if (debugConsole) { std::cout << "Update U.000: Setting the training session neural network archives directory.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error TD.000: An empty path was provided for the training session neural network archives directory.\n"; }
		abort();
	}
	path.find_last_of("\\") == path.length() - 1 ? directoryTrainingSessionNeuralNetworkArchives = path : directoryTrainingSessionNeuralNetworkArchives = path + "\\";
	if (!std::filesystem::exists(directoryTrainingSessionNeuralNetworkArchives)) {
		if (debugConsole) { std::cout << "Update U.000: Creating the training session neural network archives directory.\n"; }
		std::filesystem::create_directory(directoryTrainingSessionNeuralNetworkArchives);
		if (debugConsole) { std::cout << "Update U.000: Finished creating the training session neural network archives directory.\n"; }
	}
	if (debugConsole) { std::cout << "Update U.000: Finished setting the training session neural network archives directory.\n"; }
}
void ImageTrainingDataset::setFilePathTrainingSessionAnnotationList(const std::string& path)
{
	if (debugConsole) { std::cout << "Update U.020: Setting the training session annotation list text file path.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error TD.005: An empty path was provided for the training session annotation list text file path.\n"; }
		abort();
	}
	filePathTrainingSessionAnnotationList = path + "List of Image Annotations.txt";
	if (!std::filesystem::exists(filePathTrainingSessionAnnotationList)) {
		if (debugConsole) { std::cout << "Update U.021: Creating the training session annotation list text file.\n"; }
		std::fstream fileAnnotationList;
		fileAnnotationList.open(filePathTrainingSessionAnnotationList, std::ios::out);
		if (fileAnnotationList.fail()) {
			if (debugConsole) { std::cout << "Error TD.006: Failed to create the training session annotation list text file.\n"; }
			abort();
		}
		fileAnnotationList.close();
		if (debugConsole) { std::cout << "Update U.022: Finished creating the training session annotation list text file.\n"; }
	}
	if (debugConsole) { std::cout << "Update U.025: Finished setting the training session annotation list text file path.\n"; }
}
void ImageTrainingDataset::setFilePathTrainingSessionTrainingImagesList(const std::string& path)
{
	if (debugConsole) { std::cout << "Update U.020: Setting the training session training images list text file path.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error TD.005: An empty path was provided for the training session training images list text file path.\n"; }
		abort();
	}
	filePathTrainingSessionTrainingImagesList = path + "List of Training Images.txt";
	if (!std::filesystem::exists(filePathTrainingSessionTrainingImagesList)) {
		if (debugConsole) { std::cout << "Update U.021: Creating the training session training images list text file.\n"; }
		std::fstream fileTrainingImagesList;
		fileTrainingImagesList.open(filePathTrainingSessionTrainingImagesList, std::ios::out);
		if (fileTrainingImagesList.fail()) {
			if (debugConsole) { std::cout << "Error TD.006: Failed to create the training session training images list text file.\n"; }
			abort();
		}
		fileTrainingImagesList.close();
		if (debugConsole) { std::cout << "Update U.022: Finished creating the training session training images list text file.\n"; }
	}
	if (debugConsole) { std::cout << "Update U.025: Finished setting the training session training images list text file path.\n"; }
}
void ImageTrainingDataset::setFilePathTrainingSessionValidationImagesList(const std::string& path)
{
	if (debugConsole) { std::cout << "Update U.020: Setting the training session validation images list text file path.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error TD.005: An empty path was provided for the training session validation images list text file path.\n"; }
		abort();
	}
	filePathTrainingSessionValidationImagesList = path + "List of Validation Images.txt";
	if (!std::filesystem::exists(filePathTrainingSessionValidationImagesList)) {
		if (debugConsole) { std::cout << "Update U.021: Creating the training session validation images list text file.\n"; }
		std::fstream fileValidationImagesList;
		fileValidationImagesList.open(filePathTrainingSessionValidationImagesList, std::ios::out);
		if (fileValidationImagesList.fail()) {
			if (debugConsole) { std::cout << "Error TD.006: Failed to create the training session validation images list text file.\n"; }
			abort();
		}
		fileValidationImagesList.close();
		if (debugConsole) { std::cout << "Update U.022: Finished creating the training session validation images list text file.\n"; }
	}
	if (debugConsole) { std::cout << "Update U.025: Finished setting the training session validation images list text file path.\n"; }
}
void ImageTrainingDataset::setFilePathTrainingSessionTestImagesList(const std::string& path)
{
	if (debugConsole) { std::cout << "Update U.020: Setting the training session test images list text file path.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error TD.005: An empty path was provided for the training session test images list text file path.\n"; }
		abort();
	}
	filePathTrainingSessionTestImagesList = path + "List of Test Images.txt";
	if (!std::filesystem::exists(filePathTrainingSessionTestImagesList)) {
		if (debugConsole) { std::cout << "Update U.021: Creating the training session test images list text file.\n"; }
		std::fstream fileTestImagesList;
		fileTestImagesList.open(filePathTrainingSessionTestImagesList, std::ios::out);
		if (fileTestImagesList.fail()) {
			if (debugConsole) { std::cout << "Error TD.006: Failed to create the training session test images list text file.\n"; }
			abort();
		}
		fileTestImagesList.close();
		if (debugConsole) { std::cout << "Update U.022: Finished creating the training session test images list text file.\n"; }
	}
	if (debugConsole) { std::cout << "Update U.025: Finished setting the training session test images list text file path.\n"; }
}
void ImageTrainingDataset::setFilePathTrainingSessionDatasetConfiguration(const std::string& path)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the training session dataset configuration text file path.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error ITD.000: An empty path was provided for the training session dataset configuration text file path.\n"; }
		abort();
	}
	filePathTrainingSessionDatasetConfiguration = path + "Dataset Configuration.txt";
	if (!std::filesystem::exists(filePathTrainingSessionDatasetConfiguration)) {
		if (debugConsole) { std::cout << "Update ITD.000: Creating the training session dataset configuration text file.\n"; }
		std::fstream datasetConfigurationFile;
		datasetConfigurationFile.open(filePathTrainingSessionDatasetConfiguration, std::ios::out);
		if (datasetConfigurationFile.fail()) {
			if (debugConsole) { std::cout << "Error ITD.000: Failed to create the training session dataset configuration text file.\n"; }
			abort();
		}
		datasetConfigurationFile.close();
		if (debugConsole) { std::cout << "Update ITD.000: Finished creating the training session dataset configuration text file.\n"; }
	}
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the training session dataset configuration text file path.\n"; }
}
void ImageTrainingDataset::setTrainingSessionAnnotationList(void)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the training session annotation list.\n"; }
	std::fstream fileAnnotationList;
	fileAnnotationList.open(filePathTrainingSessionAnnotationList, std::ios::in);
	if (fileAnnotationList.fail()) {
		if (debugConsole) { std::cout << "Error ITD.000: Failed to open the training session annotation list text file.\n"; }
		abort();
	}
	trainingSessionAnnotationList.clear();
	std::string line;
	do {
		std::getline(fileAnnotationList, line);
		if (!line.empty() && !fileAnnotationList.fail()) {
			trainingSessionAnnotationList.push_back(line);
		}
	} while (!line.empty() && !fileAnnotationList.fail());
	fileAnnotationList.close();
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the training session annotation list.\n"; }
}
void ImageTrainingDataset::setTrainingSessionImageWidth(const unsigned& width)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the training session image width.\n"; }
	trainingSessionImageWidth = width;
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the training session image width.\n"; }
}
void ImageTrainingDataset::setTrainingSessionImageHeight(const unsigned& height)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the training session image height.\n"; }
	trainingSessionImageHeight = height;
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the training session image height.\n"; }
}
void ImageTrainingDataset::setTrainingSessionNeuralNetworkTopology(const std::vector<unsigned>& hiddenLayersTopology)
{
	if (debugConsole) { std::cout << "Update ITD.000: \n"; };
	trainingSessionNeuralNetworkTopology.clear();
	if (trainingSessionImageWidth == 0 || trainingSessionImageHeight == 0 || trainingSessionAnnotationList.size() == 0) {
		if (debugConsole) { std::cout << "Error ITD.000: \n"; };
		abort();
	}
	trainingSessionNeuralNetworkTopology.push_back(trainingSessionImageWidth * trainingSessionImageHeight);
	for (unsigned layerIndex = 0; layerIndex < hiddenLayersTopology.size(); layerIndex++) {
		trainingSessionNeuralNetworkTopology.push_back(hiddenLayersTopology[layerIndex]);
	}
	trainingSessionNeuralNetworkTopology.push_back(trainingSessionAnnotationList.size());
	if (debugConsole) { std::cout << "Update ITD.000: \n"; };
}
void ImageTrainingDataset::setTrainingSessionNeuralNetworkEta(const double& eta)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the training session network eta.\n"; }
	trainingSessionNeuralNetworkEta = eta;
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the training session network eta.\n"; }
}
void ImageTrainingDataset::setTrainingSessionNeuralNetworkAlpha(const double& alpha)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the training session network alpha.\n"; }
	trainingSessionNeuralNetworkAlpha = alpha;
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the training session network alpha.\n"; }
}

void ImageTrainingDataset::createNewTrainingSession(const std::string& path)
{
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
	setDirectoryCurrentTrainingSession(directoryTrainingSessions + path);
	setDirectoryTrainingSessionImages(directoryCurrentTrainingSession + "Images\\");
	setDirectoryTrainingSessionImageAnnotations(directoryTrainingSessionImages + "Image Annotations\\");
	setDirectoryTrainingSessionTargetOutputDataFiles(directoryCurrentTrainingSession + "Target Output Data Files\\");
	setDirectoryTrainingSessionNeuralNetworkArchives(directoryCurrentTrainingSession + "Neural Network Archives\\");
	setFilePathTrainingSessionAnnotationList(directoryCurrentTrainingSession + "Annotation List.txt");
	setFilePathTrainingSessionDatasetConfiguration(directoryCurrentTrainingSession + "Dataset Configuration.txt");
	updateDirectoryTrainingSessionImages();
	updateDirectoryTrainingSessionImageAnnotations();
	updateFileTrainingSessionAnnotationList();
	verifyTrainingSessionImageSizesMatch();
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
}
void ImageTrainingDataset::selectExistingTrainingSession(const std::string& path)
{
	if (debugConsole) { std::cout << "Update ITD.000: Selecting an existing training session directory.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error ITD.000: An empty path was provided for the existing training session directory.\n"; }
		abort();
	}
	std::string tempPath = directoryTrainingSessions + path;
	if (path.find_last_of("\\") != path.length() - 1) {
		tempPath += "\\";
	}
	if (!std::filesystem::exists(tempPath)) {
		abort();
	}
	setDirectoryCurrentTrainingSession(tempPath);
	setDirectoryTrainingSessionImages(directoryCurrentTrainingSession + "Images\\");
	setDirectoryTrainingSessionImageAnnotations(directoryTrainingSessionImages + "Image Annotations\\");
	setDirectoryTrainingSessionTargetOutputDataFiles(directoryCurrentTrainingSession + "Target Output Data Files\\");
	setDirectoryTrainingSessionNeuralNetworkArchives(directoryCurrentTrainingSession + "Neural Network Archives\\");
	setFilePathTrainingSessionAnnotationList(directoryCurrentTrainingSession);
	setFilePathTrainingSessionTrainingImagesList(directoryCurrentTrainingSession);
	setFilePathTrainingSessionValidationImagesList(directoryCurrentTrainingSession);
	setFilePathTrainingSessionTestImagesList(directoryCurrentTrainingSession);
	setFilePathTrainingSessionDatasetConfiguration(directoryCurrentTrainingSession);
	setTrainingSessionAnnotationList();
	verifyTrainingSessionImageSizesMatch();
	if (debugConsole) { std::cout << "Update ITD.000: Finished selecting an existing training session directory.\n"; }
}

void ImageTrainingDataset::updateDirectoryTrainingSessionImages(void)
{
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
	for (auto const& dir_entry : std::filesystem::directory_iterator(directoryImages)) {
		if (std::filesystem::is_regular_file(dir_entry)) {
			std::filesystem::path destination_path = directoryTrainingSessionImages / dir_entry.path().filename();
			if (!std::filesystem::exists(destination_path)) {
				try {
					std::filesystem::copy_file(dir_entry.path(), directoryTrainingSessionImages + dir_entry.path().filename().string());
				}
				catch (const std::filesystem::filesystem_error& e) {
					if (debugConsole) {
						std::cout << "Error ITD.000: Error copying image file:\n" << dir_entry.path().string() << std::endl;
					}
					abort();
				}
			}
		}
	}
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
}
void ImageTrainingDataset::updateDirectoryTrainingSessionImageAnnotations(void)
{
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
	for (auto const& dir_entry : std::filesystem::directory_iterator(directoryImageAnnotations)) {
		if (std::filesystem::is_regular_file(dir_entry)) {
			std::filesystem::path destination_path = directoryTrainingSessionImageAnnotations / dir_entry.path().filename();
			if (std::filesystem::exists(destination_path)) {
				try {
					std::filesystem::remove(destination_path);
				}
				catch (const std::filesystem::filesystem_error& e) {
					if (debugConsole) {
						std::cout << "Error ITD.000: Error removing the existing image annotations file:\n" << destination_path.string() << std::endl;
					}
					abort();
				}
			}
			try {
				std::filesystem::copy_file(dir_entry.path(), directoryTrainingSessionImageAnnotations + dir_entry.path().filename().string());
			}
			catch (const std::filesystem::filesystem_error& e) {
				if (debugConsole) {
					std::cout << "Error ITD.000: Error copying image annotation text file:\n" << dir_entry.path().string() << std::endl;
				}
				abort();
			}
		}
	}
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
}
void ImageTrainingDataset::updateFileTrainingSessionAnnotationList(void)
{
	if (debugConsole) { std::cout << "Update ITD.000: Updating the training session annotation list text file.\n"; }
	for (auto const& dir_entry : std::filesystem::directory_iterator(directoryTrainingSessionImageAnnotations)) {
		if (std::filesystem::is_regular_file(dir_entry) && std::filesystem::path(dir_entry).extension() == ".txt") {
			std::fstream fileImageAnnotations;
			fileImageAnnotations.open(dir_entry, std::ios::in);
			if (fileImageAnnotations.fail()) {
				if (debugConsole) { std::cout << "Error ITD.000: Failed to open a training session image annotations text file.\n"; }
				abort();
			}
			std::string annotation;
			do {
				std::getline(fileImageAnnotations, annotation);
				if (!annotation.empty() && !fileImageAnnotations.fail()) {
					std::fstream fileAnnotationList;
					fileAnnotationList.open(filePathTrainingSessionAnnotationList, std::ios::in);
					if (fileAnnotationList.fail()) {
						if (debugConsole) { std::cout << "Error ITD.000: Failed to open the training session annotation list text file.\n"; }
						abort();
					}
					std::string listAnnotation;
					bool foundAnnotation = false;
					do { // revisit this getline for fail...
						std::getline(fileAnnotationList, listAnnotation);
						if (listAnnotation == annotation) {
							foundAnnotation = true;
						}
					} while (!listAnnotation.empty() && !foundAnnotation);
					fileAnnotationList.close();
					if (!foundAnnotation) {
						fileAnnotationList.open(filePathTrainingSessionAnnotationList, std::ios::app);
						if (fileAnnotationList.fail()) {
							if (debugConsole) { std::cout << "Error ITD.000: Failed to open the training session annotation list text file.\n"; }
							abort();
						}
						fileAnnotationList << annotation << std::endl;
						fileAnnotationList.close();
					}
				}
			} while (!annotation.empty() && !fileImageAnnotations.fail());
			fileImageAnnotations.close();
		}
	}
	if (debugConsole) { std::cout << "Update ITD.000: Finished updating the training session annotations list text file.\n"; }
	setTrainingSessionAnnotationList();
}
void ImageTrainingDataset::updateTrainingSessionTargetOutputDataFiles(void)
{
	if (debugConsole) { std::cout << "Update ITD.000: Creating the training session target output data text files.\n"; }
	// Comment Here
	for (auto const& dir_entry : std::filesystem::directory_iterator(directoryTrainingSessionImageAnnotations)) {
		// Comment Here
		if (std::filesystem::is_regular_file(dir_entry) && std::filesystem::path(dir_entry).extension() == ".txt") {
			std::fstream fileImageAnnotations;
			std::string line;
			std::vector<std::string> imageAnnotations;
			fileImageAnnotations.open(dir_entry, std::ios::in);
			do {
				std::getline(fileImageAnnotations, line);
				if (!line.empty() && !fileImageAnnotations.fail()) {
					imageAnnotations.push_back(line);
				}
			} while (!line.empty() && !fileImageAnnotations.fail());
			fileImageAnnotations.close();
			// Comment Here
			std::string filePathTargetOutputData = directoryTrainingSessionTargetOutputDataFiles + std::filesystem::path(dir_entry).stem().string() + ".TOD.txt";
			std::fstream fileTargetOutputData;
			fileTargetOutputData.open(filePathTargetOutputData, std::ios::out);
			for (std::string annotationFromList : trainingSessionAnnotationList) {
				bool found = false;
				for (std::string annotationFromImage : imageAnnotations) {
					if (annotationFromList == annotationFromImage) { found = true; }
				}
				found ? fileTargetOutputData << "1\n" : fileTargetOutputData << "0\n";
			}
			// Comment Here (bias node on output layer)
			fileTargetOutputData << "1\n";
			fileTargetOutputData.close();
		}
	}
	if (debugConsole) { std::cout << "Update ITD.000: Finished creating the training session target output data text files.\n"; }
}
void ImageTrainingDataset::updateTrainingSessionTrainingValidationAndTestFiles(const double& percentTrainingData, const double& percentValidationData, const double& percentTestData)
{
	if (percentTrainingData + percentValidationData + percentTestData != 1.0) {
		abort();
	}
	unsigned numberOfImages = 0;
	for (auto const& dir_entry : std::filesystem::directory_iterator(directoryImages)) {
		if (std::filesystem::is_regular_file(dir_entry)) { numberOfImages++; }		
	}
	if (numberOfImages < 3) {
		abort();
	}
	unsigned numberOfTrainingImages = std::round(numberOfImages * percentTrainingData);
	if (numberOfTrainingImages == 0) { numberOfTrainingImages = 1; }
	unsigned numberOfValidationImages = std::round(numberOfImages * percentValidationData);
	if (numberOfValidationImages == 0) { numberOfValidationImages = 1; }
	unsigned numberOfTestImages = std::round(numberOfImages * percentTestData);
	if (numberOfTestImages == 0) { numberOfTestImages = 1; }
	unsigned sumOfImages = numberOfTrainingImages + numberOfValidationImages + numberOfTestImages;
	if (sumOfImages > numberOfImages) {
		do {
			if (numberOfTrainingImages >= 2 && sumOfImages != numberOfImages) { numberOfTrainingImages--; }
			sumOfImages = numberOfTrainingImages + numberOfValidationImages + numberOfTestImages;
			if (numberOfValidationImages >= 2 && sumOfImages != numberOfImages) { numberOfValidationImages--; }
			sumOfImages = numberOfTrainingImages + numberOfValidationImages + numberOfTestImages;
			if (numberOfTestImages >= 2 && sumOfImages != numberOfImages) {	numberOfTestImages--; }
			sumOfImages = numberOfTrainingImages + numberOfValidationImages + numberOfTestImages;
		} while (sumOfImages != numberOfImages);
	}
	else if (sumOfImages < numberOfImages) {
		do {
			if (sumOfImages != numberOfImages) { numberOfTrainingImages++; }
			sumOfImages = numberOfTrainingImages + numberOfValidationImages + numberOfTestImages;
			if (sumOfImages != numberOfImages) { numberOfValidationImages++; }
			sumOfImages = numberOfTrainingImages + numberOfValidationImages + numberOfTestImages;
			if (sumOfImages != numberOfImages) { numberOfTestImages++; }
			sumOfImages = numberOfTrainingImages + numberOfValidationImages + numberOfTestImages;
		} while (sumOfImages != numberOfImages);
	}
	std::vector<std::string> listOfTrainingImages;
	std::vector<std::string> listOfValidationImages;
	std::vector<std::string> listOfTestImages;
	for (auto const& dir_entry : std::filesystem::directory_iterator(directoryImages)) {
		if (std::filesystem::is_regular_file(dir_entry)) {
			if (listOfTrainingImages.size() < numberOfTrainingImages) { 
				listOfTrainingImages.push_back(dir_entry.path().string());
			}
			else if (listOfValidationImages.size() < numberOfValidationImages) {
				listOfValidationImages.push_back(dir_entry.path().string());
			}
			else if (listOfTestImages.size() < numberOfTestImages) {
				listOfTestImages.push_back(dir_entry.path().string());
			}
			else {
				abort();
			}
		}
	}
	std::fstream fileTrainingImagesList;
	fileTrainingImagesList.open(filePathTrainingSessionTrainingImagesList, std::ios::out);
	for (const std::string& imagePath : listOfTrainingImages) {
		fileTrainingImagesList << imagePath << std::endl;
	}
	fileTrainingImagesList.close();
	std::fstream fileValidationImagesList;
	fileValidationImagesList.open(filePathTrainingSessionValidationImagesList, std::ios::out);
	for (const std::string& imagePath : listOfValidationImages) {
		fileValidationImagesList << imagePath << std::endl;
	}
	fileValidationImagesList.close();
	std::fstream fileTestImagesList;
	fileTestImagesList.open(filePathTrainingSessionTestImagesList, std::ios::out);
	for (const std::string& imagePath : listOfTestImages) {
		fileTestImagesList << imagePath << std::endl;
	}
	fileTestImagesList.close();
}
void ImageTrainingDataset::updateTrainingSession(const double& percentTrainingData, const double& percentValidationData, const double& percentTestData)
{
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
	updateDirectoryTrainingSessionImages();
	updateDirectoryTrainingSessionImageAnnotations();
	updateFileTrainingSessionAnnotationList();
	updateTrainingSessionTargetOutputDataFiles();
	updateTrainingSessionTrainingValidationAndTestFiles(percentTrainingData, percentValidationData, percentTestData);
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
}

void ImageTrainingDataset::verifyTrainingSessionImageSizesMatch(void)
{
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
	bool updatedSize = false;
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
	setTrainingSessionImageWidth(0);
	setTrainingSessionImageHeight(0);
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
	for (auto const& dir_entry : std::filesystem::directory_iterator(directoryTrainingSessionImages)) {
		if (std::filesystem::is_regular_file(dir_entry)) {
			cv::Mat inputImage = cv::imread(dir_entry.path().string());
			if (inputImage.empty()) {
				if (debugConsole) { std::cout << "Error ITD.000: \n"; }
				abort();
			}
			if (!updatedSize) {
				if (debugConsole) { std::cout << "Update ITD.000: \n"; }
				setTrainingSessionImageWidth(inputImage.cols);
				setTrainingSessionImageHeight(inputImage.rows);
				if (debugConsole) { std::cout << "Update ITD.000: \n"; }
				updatedSize = true;
			}
			else {
				if (inputImage.cols != trainingSessionImageWidth || inputImage.cols != trainingSessionImageHeight) {
					if (debugConsole) { std::cout << "Error ITD.000: \n"; }
					abort();
				}
			}
		}
	}
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
}
void ImageTrainingDataset::updateFileTrainingSessionDatasetConfiguration(void)
{
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
	std::fstream fileDatasetConfiguration;
	fileDatasetConfiguration.open(filePathTrainingSessionDatasetConfiguration, std::ios::out);
	if (fileDatasetConfiguration.fail()) {
		if (debugConsole) { std::cout << "Error ITD.000: Failed to open the training session dataset configuration text file.\n"; }
		abort();
	}
	fileDatasetConfiguration << "Directory Image Training Dataset:\n" << directoryImageTrainingDataset << "\n\n";
	fileDatasetConfiguration << "Directory Images:\n" << directoryImages << "\n\n";
	fileDatasetConfiguration << "Directory Image Annotations:\n" << directoryImageAnnotations << "\n\n";
	fileDatasetConfiguration << "Directory Training Sessions:\n" << directoryTrainingSessions << "\n\n";
	fileDatasetConfiguration << "Directory Current Training Session:\n" << directoryCurrentTrainingSession << "\n\n";
	fileDatasetConfiguration << "Directory Training Session Images:\n" << directoryTrainingSessionImages << "\n\n";
	fileDatasetConfiguration << "Directory Training Session Image Annotations:\n" << directoryTrainingSessionImageAnnotations << "\n\n";
	fileDatasetConfiguration << "Directory Training Session Target Output Data Files:\n" << directoryTrainingSessionTargetOutputDataFiles << "\n\n";
	fileDatasetConfiguration << "Directory Training Session Neural Network Archives:\n" << directoryTrainingSessionNeuralNetworkArchives << "\n\n";
	fileDatasetConfiguration << "File Path Training Session Annotation List:\n" << filePathTrainingSessionAnnotationList << "\n\n";
	fileDatasetConfiguration << "File Path Training Session Training Images List:\n" << filePathTrainingSessionTrainingImagesList << "\n\n";
	fileDatasetConfiguration << "File Path Training Session Validation Images List:\n" << filePathTrainingSessionValidationImagesList << "\n\n";
	fileDatasetConfiguration << "File Path Training Session Test Images List:\n" << filePathTrainingSessionTestImagesList << "\n\n";
	fileDatasetConfiguration << "Training Session Width Of Input Images: " << trainingSessionImageWidth << std::endl;
	fileDatasetConfiguration << "Training Session Height Of Input Images: " << trainingSessionImageHeight << std::endl;
	fileDatasetConfiguration << "Training Session Number Of Image Annotations: " << trainingSessionAnnotationList.size() << "\n\n";
	fileDatasetConfiguration << "Training Session Neural Network Input Layer Size: " << trainingSessionImageWidth * trainingSessionImageHeight + 1 << std::endl;
	fileDatasetConfiguration << "Training Session Neural Network Output Layer Size: " << trainingSessionAnnotationList.size() + 1 << std::endl;
	fileDatasetConfiguration << "Training Session Neural Network Eta: " << trainingSessionNeuralNetworkEta << std::endl;
	fileDatasetConfiguration << "Training Session Neural Network Alpha: " << trainingSessionNeuralNetworkAlpha << std::endl;
	fileDatasetConfiguration << "Training Session Neural Network Topology: ";
	for (unsigned topologyIndex = 0; topologyIndex < trainingSessionNeuralNetworkTopology.size(); topologyIndex++) {
		fileDatasetConfiguration << trainingSessionNeuralNetworkTopology[topologyIndex] + 1 << (topologyIndex != trainingSessionNeuralNetworkTopology.size() - 1 ? "," : "\n");
	}
	fileDatasetConfiguration.close();
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
}
void ImageTrainingDataset::prepareTrainingSession(const double& eta, const double& alpha, const std::vector<unsigned>& hiddenLayersTopology)
{
	if (debugConsole) { std::cout << "Update ITD.000: Preparing the image training dataset for training.\n"; }
	verifyTrainingSessionImageSizesMatch();
	setTrainingSessionNeuralNetworkEta(eta);
	setTrainingSessionNeuralNetworkAlpha(alpha);
	setTrainingSessionNeuralNetworkTopology(hiddenLayersTopology);	
	updateFileTrainingSessionDatasetConfiguration();
	if (debugConsole) { std::cout << "Update ITD.000: Finished preparing the image training dataset for training.\n"; }
}

void ImageTrainingDataset::createNewNeuralNetwork(void)
{	
	trainingSessionNeuralNetwork.createNewNeuralNetwork(trainingSessionNeuralNetworkTopology, trainingSessionNeuralNetworkEta, trainingSessionNeuralNetworkAlpha);
}
void ImageTrainingDataset::loadExistingNeuralNetwork(const std::string& fileStem)
{
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
	if (fileStem.empty()) {
		if (debugConsole) { std::cout << "Error ITD.000: \n"; }
		abort();
	}
	std::string filePathArchive = directoryTrainingSessionNeuralNetworkArchives + fileStem + ".txt";
	if (!std::filesystem::exists(filePathArchive)) {
		if (debugConsole) { std::cout << "Error ITD.000: \n"; }
		abort();
	}
	trainingSessionNeuralNetwork.loadExistingNeuralNetwork(trainingSessionNeuralNetworkTopology, filePathArchive);
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
}
void ImageTrainingDataset::archiveNeuralNetwork(const std::string& fileStem)
{
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
	if (fileStem.empty()) {
		if (debugConsole) { std::cout << "Error ITD.000: \n"; }
		abort();
	}
	std::string filePathArchive = directoryTrainingSessionNeuralNetworkArchives + fileStem + ".txt";
	trainingSessionNeuralNetwork.archiveNeuralNetwork(trainingSessionNeuralNetworkTopology, filePathArchive);
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
}

// Working On These Three Functions Currently
void ImageTrainingDataset::runTrainingImages(void)
{

}
void ImageTrainingDataset::runValidationImages(void)
{

}
void ImageTrainingDataset::runTestImages(void)
{

}
