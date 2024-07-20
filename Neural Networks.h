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
	networkError = std::sqrt(networkError); // RMS

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
	// Private Variables - Image Training Dataset (Encapsulation...)
	bool debugConsole;

	std::string directoryImageTrainingDataset;

	std::string directoryImages;
	std::string directoryImageAnnotations;

	std::string directoryTrainingSessions;
	std::string directoryCurrentTrainingSession;
	std::string directoryTrainingSessionImages;
	std::string directoryTrainingSessionImageAnnotations;
	std::string directoryTrainingSessionInputDataFiles;
	std::string directoryTrainingSessionTargetOutputDataFiles;
	std::string directoryTrainingSessionNeuralNetworkArchives;

	std::string filePathTrainingSessionListImageAnnotations;
	std::string filePathTrainingSessionListTrainingImages;
	std::string filePathTrainingSessionListValidationImages;
	std::string filePathTrainingSessionListTestImages;

	std::string filePathTrainingSessionDatasetConfiguration;

	std::vector<std::string> trainingSessionListImageAnnotations;
	std::vector<std::string> trainingSessionListTrainingImages;
	std::vector<std::string> trainingSessionListValidationImages;
	std::vector<std::string> trainingSessionListTestImages;

	unsigned trainingSessionImageWidth;
	unsigned trainingSessionImageHeight;

	// Private Variables - Neural Network (Encapsulation...)
	NeuralNetwork trainingSessionNeuralNetwork;

	double trainingSessionNeuralNetworkEta;
	double trainingSessionNeuralNetworkAlpha;
	std::vector<unsigned> trainingSessionNeuralNetworkTopology;

public:
	// Public Constructors
	ImageTrainingDataset(const std::string& path) { initializeImageTrainingDataset(true, path); };
	ImageTrainingDataset(const std::string& path, const std::string& session) { initializeImageTrainingDataset(true, path, session); };
	ImageTrainingDataset(const bool& console, const std::string& path) { initializeImageTrainingDataset(console, path); };
	ImageTrainingDataset(const bool& console, const std::string& path, const std::string& session) { initializeImageTrainingDataset(console, path, session); };

	// Public Setters - Image Training Dataset
	void setDebugConsole(const bool& console);

	void setDirectoryImages(const std::string& path);
	void setDirectoryImageAnnotations(const std::string& path);

	// Public Functions - Image Training Dataset
	void newTrainingSession(const std::string& path);
	void existingTrainingSession(const std::string& path);
	void updateTrainingSession(const double& percentTrainingData, const double& percentValidationData, const double& percentTestData);		

	// Public Functions - Neural Network
	void newNeuralNetwork(const double& eta, const double& alpha, const std::vector<unsigned>& hiddenLayersTopology);
	void existingNeuralNetwork(const std::string& fileStem);
	void archiveNeuralNetwork(const std::string& fileStem);

	void runTrainingImages(void);
	void runValidationImages(void); // Currently An Empty Function... Will Come Back To Later...
	void runTestImages(void); // Currently An Empty Function... Will Come Back To Later...

	// Public Getters - Image Training Dataset
	bool getDebugConsole(void) const { return debugConsole; };

	std::string getDirectoryImageTrainingDataset(void) const { return directoryImageTrainingDataset; };

	std::string getDirectoryImages(void) const { return directoryImages; };
	std::string getDirectoryImageAnnotations(void) const { return directoryImageAnnotations; };

	std::string getDirectoryTrainingSessions(void) const { return directoryTrainingSessions; };
	std::string getDirectoryCurrentTrainingSession(void) const { return directoryCurrentTrainingSession; };
	std::string getDirectoryTrainingSessionImages(void) const { return directoryTrainingSessionImages; };
	std::string getDirectoryTrainingSessionImageAnnotations(void) const { return directoryTrainingSessionImageAnnotations; };
	std::string getDirectoryTrainingSessionInputDataFiles(void) const { return directoryTrainingSessionInputDataFiles; };
	std::string getDirectoryTrainingSessionTargetOutputDataFiles(void) const { return directoryTrainingSessionTargetOutputDataFiles; };
	std::string getDirectoryTrainingSessionNeuralNetworkArchives(void) const { return directoryTrainingSessionNeuralNetworkArchives; };

	std::string getFilePathTrainingSessionListImageAnnotations(void) const { return filePathTrainingSessionListImageAnnotations; };
	std::string getFilePathTrainingSessionListTrainingImages(void) const { return filePathTrainingSessionListTrainingImages; };
	std::string getFilePathTrainingSessionListValidationImages(void) const { return filePathTrainingSessionListValidationImages; };
	std::string getFilePathTrainingSessionListTestImages(void) const { return filePathTrainingSessionListTestImages; };

	std::string getFilePathTrainingSessionDatasetConfiguration(void) const { return filePathTrainingSessionDatasetConfiguration; };

	std::vector<std::string> getTrainingSessionListImageAnnotations(void) const { return trainingSessionListImageAnnotations; };
	std::vector<std::string> getTrainingSessionListTrainingImages(void) const { return trainingSessionListTrainingImages; };
	std::vector<std::string> getTrainingSessionListValidationImages(void) const { return trainingSessionListValidationImages; };
	std::vector<std::string> getTrainingSessionListTestImages(void) const { return trainingSessionListTestImages; };

	unsigned getTrainingSessionImageWidth(void) const { return trainingSessionImageWidth; };
	unsigned getTrainingSessionImageHeight(void) const { return trainingSessionImageHeight; };

	// Public Getters - Neural Network
	double getTrainingSessionNeuralNetworkEta(void) const { return trainingSessionNeuralNetworkEta; };
	double getTrainingSessionNeuralNetworkAlpha(void) const { return trainingSessionNeuralNetworkAlpha; };
	std::vector<unsigned> getTrainingSessionNeuralNetworkTopology(void) const { return trainingSessionNeuralNetworkTopology; };	

private:
	// Private Functions - Image Training Dataset (Could Be Public... Your Choice... Your Risk...)
	void initializeImageTrainingDataset(const bool& console, const std::string& path);
	void initializeImageTrainingDataset(const bool& console, const std::string& path, const std::string& session);

	void initializeTrainingSessionNeuralNetwork(void);

	void updateDirectoryTrainingSessionImages(void);
	void verifyTrainingSessionImageSizesMatch(void);
	void updateDirectoryTrainingSessionImageAnnotations(void);
	void verifyTrainingSessionImagesHaveImageAnnotations(void);
	void updateFileTrainingSessionListImageAnnotations(void);
	void updateTrainingSessionInputDataFiles(void);
	void updateTrainingSessionTargetOutputDataFiles(void);
	void updateTrainingSessionTrainingValidationAndTestFiles(const double& percentTrainingData, const double& percentValidationData, const double& percentTestData);

	void updateFileTrainingSessionDatasetConfiguration(void);

	// Private Setters - Image Training Dataset (Could Be Public... Your Choice... Your Risk...)
	void setDirectoryImageTrainingDataset(const std::string& path);
	void setDirectoryTrainingSessions(const std::string& path);

	void setDirectoryCurrentTrainingSession(const std::string& path);
	void setDirectoryTrainingSessionImages(const std::string& path);
	void setDirectoryTrainingSessionImageAnnotations(const std::string& path);
	void setDirectoryTrainingSessionInputDataFiles(const std::string& path);
	void setDirectoryTrainingSessionTargetOutputDataFiles(const std::string& path);
	void setDirectoryTrainingSessionNeuralNetworkArchives(const std::string& path);

	void setFilePathTrainingSessionListImageAnnotations(const std::string& path);
	void setFilePathTrainingSessionListTrainingImages(const std::string& path);
	void setFilePathTrainingSessionListValidationImages(const std::string& path);
	void setFilePathTrainingSessionListTestImages(const std::string& path);

	void setFilePathTrainingSessionDatasetConfiguration(const std::string& path);

	void setTrainingSessionListImageAnnotations(void);
	void setTrainingSessionListTrainingImages(void);
	void setTrainingSessionListValidationImages(void);
	void setTrainingSessionListTestImages(void);

	void setTrainingSessionImageWidth(const unsigned& width);
	void setTrainingSessionImageHeight(const unsigned& height);

	// Private Setters - Neural Network (Could Be Public... Your Choice... Your Risk...)
	void setTrainingSessionNeuralNetworkEta(const double& eta);
	void setTrainingSessionNeuralNetworkAlpha(const double& alpha);
	void setTrainingSessionNeuralNetworkTopology(const std::vector<unsigned>& hiddenLayersTopology);
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
	setDirectoryTrainingSessionInputDataFiles(directoryCurrentTrainingSession + "Input Data Files\\");
	setDirectoryTrainingSessionTargetOutputDataFiles(directoryCurrentTrainingSession + "Target Output Data Files\\");
	setDirectoryTrainingSessionNeuralNetworkArchives(directoryCurrentTrainingSession + "Neural Network Archives\\");
	setFilePathTrainingSessionListImageAnnotations(directoryCurrentTrainingSession);
	setFilePathTrainingSessionListTrainingImages(directoryCurrentTrainingSession);
	setFilePathTrainingSessionListValidationImages(directoryCurrentTrainingSession);
	setFilePathTrainingSessionListTestImages(directoryCurrentTrainingSession);
	setFilePathTrainingSessionDatasetConfiguration(directoryCurrentTrainingSession);	
	setTrainingSessionListImageAnnotations();
	setTrainingSessionListTrainingImages();
	setTrainingSessionListValidationImages();
	setTrainingSessionListTestImages();
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
	existingTrainingSession(session);
	initializeTrainingSessionNeuralNetwork();
	updateFileTrainingSessionDatasetConfiguration();
	if (debugConsole) { std::cout << "Update ITD.000: Finished initializing the image training dataset.\n"; }
}

void ImageTrainingDataset::initializeTrainingSessionNeuralNetwork(void)
{
	if (debugConsole) { std::cout << "Update ITD.000: \n"; };
	std::fstream fileDatasetConfiguration;
	fileDatasetConfiguration.open(filePathTrainingSessionDatasetConfiguration, std::ios::in);
	if (!fileDatasetConfiguration.is_open()) {
		if (debugConsole) { std::cout << "Error ITD.000: \n"; };
		abort();
	}
	std::string line;
	std::getline(fileDatasetConfiguration, line);
	if (fileDatasetConfiguration.fail() || line.empty()) {
		// Set Default Values
		trainingSessionNeuralNetworkEta = 0.15;
		trainingSessionNeuralNetworkAlpha = 0.5;
		trainingSessionNeuralNetworkTopology.clear();
	}
	else {
		// Get Neural Network Eta From File
		for (unsigned index = 0; index < 48; index++) {
			std::getline(fileDatasetConfiguration, line);
			if (fileDatasetConfiguration.fail()) {
				if (debugConsole) { std::cout << "Error ITD.000: \n"; }
				abort();
			}
		}
		line.erase(0, 37);
		double value = std::stod(line);
		value == 0 ? trainingSessionNeuralNetworkEta = 0.15 : trainingSessionNeuralNetworkEta = value;
		// Get Neural Network Alpha From File
		std::getline(fileDatasetConfiguration, line);
		if (fileDatasetConfiguration.fail() || line.empty()) {
			if (debugConsole) { std::cout << "Error ITD.000: \n"; }
			abort();
		}
		line.erase(0, 39);
		value = std::stod(line);
		value == 0 ? trainingSessionNeuralNetworkAlpha = 0.5 : trainingSessionNeuralNetworkAlpha = value;
		// Get Neural Network Topology From File
		trainingSessionNeuralNetworkTopology.clear();
		std::getline(fileDatasetConfiguration, line);
		if (fileDatasetConfiguration.fail() || line.empty()) {
			if (debugConsole) { std::cout << "Error ITD.000: \n"; }
			abort();
		}
		line.erase(0, 42);
		if (line.size() != 0) {
			bool getValue = true;
			do {
				unsigned index = line.find(",");
				if (index != -1) {
					std::string valueString = line.substr(0, index);
					line.erase(0, index + 1);
					unsigned valueUnsigned = std::stoi(valueString) - 1;
					trainingSessionNeuralNetworkTopology.push_back(valueUnsigned);
				}
				else {
					unsigned valueUnsigned = std::stoi(line) - 1;
					trainingSessionNeuralNetworkTopology.push_back(valueUnsigned);
					getValue = false;
				}
			} while (getValue);
		}
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
	if (debugConsole) { std::cout << "Update ITD.000: Setting the image training dataset directory.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error ITD.000: An empty path was provided for the image training dataset directory.\n"; }
		abort();
	}
	path.find_last_of("\\") == path.length() - 1 ? directoryImageTrainingDataset = path : directoryImageTrainingDataset = path + "\\";
	if (!std::filesystem::exists(directoryImageTrainingDataset)) {
		if (debugConsole) { std::cout << "Update ITD.000: Creating the image training dataset directory.\n"; }
		std::filesystem::create_directory(directoryImageTrainingDataset);
		if (debugConsole) { std::cout << "Update ITD.000: Finished creating the image training dataset directory.\n"; }
	}
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the image training dataset directory.\n"; }
}

void ImageTrainingDataset::setDirectoryImages(const std::string& path)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the images directory.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error ITD.000: An empty path was provided for the images directory.\n"; }
		abort();
	}
	path.find_last_of("\\") == path.length() - 1 ? directoryImages = path : directoryImages = path + "\\";
	if (!std::filesystem::exists(directoryImages)) {
		if (debugConsole) { std::cout << "Update ITD.000: Creating the images directory.\n"; }
		std::filesystem::create_directory(directoryImages);
		if (debugConsole) { std::cout << "Update ITD.000: Finished creating the images directory.\n"; }
	}
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the images directory.\n"; }
}
void ImageTrainingDataset::setDirectoryImageAnnotations(const std::string& path)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the image annotations directory.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error ITD.000: An empty path was provided for the image annotations directory.\n"; }
		abort();
	}
	path.find_last_of("\\") == path.length() - 1 ? directoryImageAnnotations = path : directoryImageAnnotations = path + "\\";
	if (!std::filesystem::exists(directoryImageAnnotations)) {
		if (debugConsole) { std::cout << "Update ITD.000: Creating the image annotations directory.\n"; }
		std::filesystem::create_directory(directoryImageAnnotations);
		if (debugConsole) { std::cout << "Update ITD.000: Finished creating the image annotations directory.\n"; }
	}
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the image annotations directory.\n"; }
}

void ImageTrainingDataset::setDirectoryTrainingSessions(const std::string& path)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the training sessions directory.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error ITD.000: An empty path was provided for the training sessions directory.\n"; }
		abort();
	}
	path.find_last_of("\\") == path.length() - 1 ? directoryTrainingSessions = path : directoryTrainingSessions = path + "\\";
	if (!std::filesystem::exists(directoryTrainingSessions)) {
		if (debugConsole) { std::cout << "Update ITD.000: Creating the training sessions directory.\n"; }
		std::filesystem::create_directory(directoryTrainingSessions);
		if (debugConsole) { std::cout << "Update ITD.000: Finished creating the training sessions directory.\n"; }
	}
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the training sessions directory.\n"; }
}
void ImageTrainingDataset::setDirectoryCurrentTrainingSession(const std::string& path)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the current training session directory.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error ITD.000: An empty path was provided for the current training session directory.\n"; }
		abort();
	}
	path.find_last_of("\\") == path.length() - 1 ? directoryCurrentTrainingSession = path : directoryCurrentTrainingSession = path + "\\";
	if (!std::filesystem::exists(directoryCurrentTrainingSession)) {
		if (debugConsole) { std::cout << "Update ITD.000: Creating a new training session directory.\n"; }
		std::filesystem::create_directory(directoryCurrentTrainingSession);
		if (debugConsole) { std::cout << "Update ITD.000: Finished creating a new training session directory.\n"; }
	}
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the current training session directory.\n"; }
}
void ImageTrainingDataset::setDirectoryTrainingSessionImages(const std::string& path)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the training session images directory.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error ITD.000: An empty path was provided for the training session images directory.\n"; }
		abort();
	}
	path.find_last_of("\\") == path.length() - 1 ? directoryTrainingSessionImages = path : directoryTrainingSessionImages = path + "\\";
	if (!std::filesystem::exists(directoryTrainingSessionImages)) {
		if (debugConsole) { std::cout << "Update ITD.000: Creating the training session images directory.\n"; }
		std::filesystem::create_directory(directoryTrainingSessionImages);
		if (debugConsole) { std::cout << "Update ITD.000: Finished creating the training session images directory.\n"; }
	}
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the training session images directory.\n"; }
}
void ImageTrainingDataset::setDirectoryTrainingSessionImageAnnotations(const std::string& path)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the training session image annotations directory.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error ITD.000: An empty path was provided for the training session image annotations directory.\n"; }
		abort();
	}
	path.find_last_of("\\") == path.length() - 1 ? directoryTrainingSessionImageAnnotations = path : directoryTrainingSessionImageAnnotations = path + "\\";
	if (!std::filesystem::exists(directoryTrainingSessionImageAnnotations)) {
		if (debugConsole) { std::cout << "Update ITD.000: Creating the training session image annotations directory.\n"; }
		std::filesystem::create_directory(directoryTrainingSessionImageAnnotations);
		if (debugConsole) { std::cout << "Update ITD.000: Finished creating the training session image annotations directory.\n"; }
	}
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the training session image annotations directory.\n"; }
}
void ImageTrainingDataset::setDirectoryTrainingSessionInputDataFiles(const std::string& path)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the training session input data files directory.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error ITD.000: An empty path was provided for the training session input data files directory.\n"; }
		abort();
	}
	path.find_last_of("\\") == path.length() - 1 ? directoryTrainingSessionInputDataFiles = path : directoryTrainingSessionInputDataFiles = path + "\\";
	if (!std::filesystem::exists(directoryTrainingSessionInputDataFiles)) {
		if (debugConsole) { std::cout << "Update ITD.000: Creating the training session input data files directory.\n"; }
		std::filesystem::create_directory(directoryTrainingSessionInputDataFiles);
		if (debugConsole) { std::cout << "Update ITD.000: Finished creating the training session input data files directory.\n"; }
	}
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the training session input data files directory.\n"; }
}
void ImageTrainingDataset::setDirectoryTrainingSessionTargetOutputDataFiles(const std::string& path)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the training session target output data files directory.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error ITD.000: An empty path was provided for the training session target output data files directory.\n"; }
		abort();
	}
	path.find_last_of("\\") == path.length() - 1 ? directoryTrainingSessionTargetOutputDataFiles = path : directoryTrainingSessionTargetOutputDataFiles = path + "\\";
	if (!std::filesystem::exists(directoryTrainingSessionTargetOutputDataFiles)) {
		if (debugConsole) { std::cout << "Update ITD.000: Creating the training session target output data files directory.\n"; }
		std::filesystem::create_directory(directoryTrainingSessionTargetOutputDataFiles);
		if (debugConsole) { std::cout << "Update ITD.000: Finished creating the training session target output data files directory.\n"; }
	}
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the training session target output data files directory.\n"; }
}
void ImageTrainingDataset::setDirectoryTrainingSessionNeuralNetworkArchives(const std::string& path)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the training session neural network archives directory.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error ITD.000: An empty path was provided for the training session neural network archives directory.\n"; }
		abort();
	}
	path.find_last_of("\\") == path.length() - 1 ? directoryTrainingSessionNeuralNetworkArchives = path : directoryTrainingSessionNeuralNetworkArchives = path + "\\";
	if (!std::filesystem::exists(directoryTrainingSessionNeuralNetworkArchives)) {
		if (debugConsole) { std::cout << "Update ITD.000: Creating the training session neural network archives directory.\n"; }
		std::filesystem::create_directory(directoryTrainingSessionNeuralNetworkArchives);
		if (debugConsole) { std::cout << "Update ITD.000: Finished creating the training session neural network archives directory.\n"; }
	}
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the training session neural network archives directory.\n"; }
}

void ImageTrainingDataset::setFilePathTrainingSessionListImageAnnotations(const std::string& path)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the training session annotation list text file path.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error ITD.000: An empty path was provided for the training session annotation list text file path.\n"; }
		abort();
	}
	filePathTrainingSessionListImageAnnotations = path + "List of Image Annotations.txt";
	if (!std::filesystem::exists(filePathTrainingSessionListImageAnnotations)) {
		if (debugConsole) { std::cout << "Update ITD.000: Creating the training session annotation list text file.\n"; }
		std::fstream fileAnnotationList;
		fileAnnotationList.open(filePathTrainingSessionListImageAnnotations, std::ios::out);
		if (fileAnnotationList.fail()) {
			if (debugConsole) { std::cout << "Error ITD.000: Failed to create the training session annotation list text file.\n"; }
			abort();
		}
		fileAnnotationList.close();
		if (debugConsole) { std::cout << "Update ITD.000: Finished creating the training session annotation list text file.\n"; }
	}
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the training session annotation list text file path.\n"; }
}
void ImageTrainingDataset::setFilePathTrainingSessionListTrainingImages(const std::string& path)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the training session training images list text file path.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error ITD.000: An empty path was provided for the training session training images list text file path.\n"; }
		abort();
	}
	filePathTrainingSessionListTrainingImages = path + "List of Training Images.txt";
	if (!std::filesystem::exists(filePathTrainingSessionListTrainingImages)) {
		if (debugConsole) { std::cout << "Update ITD.000: Creating the training session training images list text file.\n"; }
		std::fstream fileTrainingImagesList;
		fileTrainingImagesList.open(filePathTrainingSessionListTrainingImages, std::ios::out);
		if (fileTrainingImagesList.fail()) {
			if (debugConsole) { std::cout << "Error ITD.000: Failed to create the training session training images list text file.\n"; }
			abort();
		}
		fileTrainingImagesList.close();
		if (debugConsole) { std::cout << "Update ITD.000: Finished creating the training session training images list text file.\n"; }
	}
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the training session training images list text file path.\n"; }
}
void ImageTrainingDataset::setFilePathTrainingSessionListValidationImages(const std::string& path)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the training session validation images list text file path.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error ITD.000: An empty path was provided for the training session validation images list text file path.\n"; }
		abort();
	}
	filePathTrainingSessionListValidationImages = path + "List of Validation Images.txt";
	if (!std::filesystem::exists(filePathTrainingSessionListValidationImages)) {
		if (debugConsole) { std::cout << "Update ITD.000: Creating the training session validation images list text file.\n"; }
		std::fstream fileValidationImagesList;
		fileValidationImagesList.open(filePathTrainingSessionListValidationImages, std::ios::out);
		if (fileValidationImagesList.fail()) {
			if (debugConsole) { std::cout << "Error ITD.00: Failed to create the training session validation images list text file.\n"; }
			abort();
		}
		fileValidationImagesList.close();
		if (debugConsole) { std::cout << "Update ITD.000: Finished creating the training session validation images list text file.\n"; }
	}
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the training session validation images list text file path.\n"; }
}
void ImageTrainingDataset::setFilePathTrainingSessionListTestImages(const std::string& path)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the training session test images list text file path.\n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error ITD.000: An empty path was provided for the training session test images list text file path.\n"; }
		abort();
	}
	filePathTrainingSessionListTestImages = path + "List of Test Images.txt";
	if (!std::filesystem::exists(filePathTrainingSessionListTestImages)) {
		if (debugConsole) { std::cout << "Update ITD.0OO: Creating the training session test images list text file.\n"; }
		std::fstream fileTestImagesList;
		fileTestImagesList.open(filePathTrainingSessionListTestImages, std::ios::out);
		if (fileTestImagesList.fail()) {
			if (debugConsole) { std::cout << "Error ITD.000: Failed to create the training session test images list text file.\n"; }
			abort();
		}
		fileTestImagesList.close();
		if (debugConsole) { std::cout << "Update ITD.000: Finished creating the training session test images list text file.\n"; }
	}
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the training session test images list text file path.\n"; }
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

void ImageTrainingDataset::setTrainingSessionListImageAnnotations(void)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the training session annotation list.\n"; }
	std::fstream fileListImageAnnotations;
	fileListImageAnnotations.open(filePathTrainingSessionListImageAnnotations, std::ios::in);
	if (fileListImageAnnotations.fail()) {
		if (debugConsole) { std::cout << "Error ITD.000: Failed to open the training session annotation list text file.\n"; }
		abort();
	}
	trainingSessionListImageAnnotations.clear();
	std::string line;
	do {
		std::getline(fileListImageAnnotations, line);
		if (!line.empty() && !fileListImageAnnotations.fail()) {
			trainingSessionListImageAnnotations.push_back(line);
		}
	} while (!line.empty() && !fileListImageAnnotations.fail());
	fileListImageAnnotations.close();
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the training session annotation list.\n"; }
}
void ImageTrainingDataset::setTrainingSessionListTrainingImages(void)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the training session training images list.\n"; }
	std::fstream fileListTrainingImages;
	fileListTrainingImages.open(filePathTrainingSessionListTrainingImages, std::ios::in);
	if (fileListTrainingImages.fail()) {
		if (debugConsole) { std::cout << "Error ITD.000: Failed to open the training session training images list text file.\n"; }
		abort();
	}
	trainingSessionListTrainingImages.clear();
	std::string line;
	do {
		std::getline(fileListTrainingImages, line);
		if (!line.empty() && !fileListTrainingImages.fail()) {
			trainingSessionListTrainingImages.push_back(line);
		}
	} while (!line.empty() && !fileListTrainingImages.fail());
	fileListTrainingImages.close();
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the training session training images list.\n"; }
}
void ImageTrainingDataset::setTrainingSessionListValidationImages(void)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the training session validation images list.\n"; }
	std::fstream fileListValidationImages;
	fileListValidationImages.open(filePathTrainingSessionListValidationImages, std::ios::in);
	if (fileListValidationImages.fail()) {
		if (debugConsole) { std::cout << "Error ITD.000: Failed to open the training session validation images list text file.\n"; }
		abort();
	}
	trainingSessionListValidationImages.clear();
	std::string line;
	do {
		std::getline(fileListValidationImages, line);
		if (!line.empty() && !fileListValidationImages.fail()) {
			trainingSessionListValidationImages.push_back(line);
		}
	} while (!line.empty() && !fileListValidationImages.fail());
	fileListValidationImages.close();
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the training session validation images list.\n"; }
}
void ImageTrainingDataset::setTrainingSessionListTestImages(void)
{
	if (debugConsole) { std::cout << "Update ITD.000: Setting the training session test images list.\n"; }
	std::fstream fileListTestImages;
	fileListTestImages.open(filePathTrainingSessionListTestImages, std::ios::in);
	if (fileListTestImages.fail()) {
		if (debugConsole) { std::cout << "Error ITD.000: Failed to open the training session test images list text file.\n"; }
		abort();
	}
	trainingSessionListTestImages.clear();
	std::string line;
	do {
		std::getline(fileListTestImages, line);
		if (!line.empty() && !fileListTestImages.fail()) {
			trainingSessionListTestImages.push_back(line);
		}
	} while (!line.empty() && !fileListTestImages.fail());
	fileListTestImages.close();
	if (debugConsole) { std::cout << "Update ITD.000: Finished setting the training session test images list.\n"; }
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
void ImageTrainingDataset::setTrainingSessionNeuralNetworkTopology(const std::vector<unsigned>& hiddenLayersTopology)
{
	if (debugConsole) { std::cout << "Update ITD.000: \n"; };
	trainingSessionNeuralNetworkTopology.clear();
	if (trainingSessionImageWidth == 0 || trainingSessionImageHeight == 0 || trainingSessionListImageAnnotations.size() == 0) {
		if (debugConsole) { std::cout << "Error ITD.000: \n"; };
		abort();
	}
	trainingSessionNeuralNetworkTopology.push_back(trainingSessionImageWidth * trainingSessionImageHeight);
	for (unsigned layerIndex = 0; layerIndex < hiddenLayersTopology.size(); layerIndex++) {
		trainingSessionNeuralNetworkTopology.push_back(hiddenLayersTopology[layerIndex]);
	}
	trainingSessionNeuralNetworkTopology.push_back(trainingSessionListImageAnnotations.size());
	if (debugConsole) { std::cout << "Update ITD.000: \n"; };
}

void ImageTrainingDataset::newTrainingSession(const std::string& path)
{
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
	if (path.empty()) {
		if (debugConsole) { std::cout << "Error ITD.000: An empty path was provided for the new training session directory.\n"; }
		abort();
	}
	std::string tempPath = directoryTrainingSessions + path;
	if (path.find_last_of("\\") != path.length() - 1) {
		tempPath += "\\";
	}
	if (std::filesystem::exists(tempPath)) {
		if (debugConsole) { std::cout << "Error ITD.000: \n"; }
		abort();
	}
	setDirectoryImages(directoryImageTrainingDataset + "Images\\");
	setDirectoryImageAnnotations(directoryImages + "Image Annotations\\");
	setDirectoryCurrentTrainingSession(tempPath);
	setDirectoryTrainingSessionImages(directoryCurrentTrainingSession + "Images\\");
	setDirectoryTrainingSessionImageAnnotations(directoryTrainingSessionImages + "Image Annotations\\");
	setDirectoryTrainingSessionInputDataFiles(directoryCurrentTrainingSession + "Input Data Files\\");
	setDirectoryTrainingSessionTargetOutputDataFiles(directoryCurrentTrainingSession + "Target Output Data Files\\");
	setDirectoryTrainingSessionNeuralNetworkArchives(directoryCurrentTrainingSession + "Neural Network Archives\\");
	setFilePathTrainingSessionListTrainingImages(directoryCurrentTrainingSession);
	setFilePathTrainingSessionListValidationImages(directoryCurrentTrainingSession);
	setFilePathTrainingSessionListTestImages(directoryCurrentTrainingSession);
	setFilePathTrainingSessionDatasetConfiguration(directoryCurrentTrainingSession);
	setTrainingSessionListImageAnnotations();
	setTrainingSessionListTrainingImages();
	setTrainingSessionListValidationImages();
	setTrainingSessionListTestImages();
	verifyTrainingSessionImageSizesMatch();
	initializeTrainingSessionNeuralNetwork();
	updateFileTrainingSessionDatasetConfiguration();
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
}
void ImageTrainingDataset::existingTrainingSession(const std::string& path)
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
		if (debugConsole) { std::cout << "Error ITD.000: The existing training session directory does not exist.\n"; }
		abort();
	}
	setDirectoryCurrentTrainingSession(tempPath);
	setFilePathTrainingSessionDatasetConfiguration(directoryCurrentTrainingSession);
	// Add Comment Here Later
	std::fstream fileDatasetConfiguration;
	fileDatasetConfiguration.open(filePathTrainingSessionDatasetConfiguration, std::ios::in);
	if (!fileDatasetConfiguration.is_open()) {
		if (debugConsole) { std::cout << "Error ITD.000: \n"; };
		abort();
	}	
	std::string line;
	// Get Main Directory For Images
	for (unsigned index = 0; index < 5; index++) {
		std::getline(fileDatasetConfiguration, line);
		if (fileDatasetConfiguration.fail()) {
			if (debugConsole) { std::cout << "Error ITD.000: \n"; }
			abort();
		}
	}
	if (line.size() == 0) {
		if (debugConsole) { std::cout << "Error ITD.000: \n"; }
		abort();
	}
	else {
		if (!std::filesystem::exists(line) || !std::filesystem::is_directory(line)) {
			if (debugConsole) { std::cout << "Error ITD.000: \n"; }
			abort();
		}
		setDirectoryImages(line);
	}
	// Get Main Directory For Image Annotations
	for (unsigned index = 0; index < 3; index++) {
		std::getline(fileDatasetConfiguration, line);
		if (fileDatasetConfiguration.fail()) {
			if (debugConsole) { std::cout << "Error ITD.000: \n"; }
			abort();
		}
	}
	if (line.size() == 0) {
		if (debugConsole) { std::cout << "Error ITD.000: \n"; }
		abort();
	}
	else {
		if (!std::filesystem::exists(line) || !std::filesystem::is_directory(line)) {
			if (debugConsole) { std::cout << "Error ITD.000: \n"; }
			abort();
		}
		setDirectoryImageAnnotations(line);
	}
	fileDatasetConfiguration.close();
	setDirectoryTrainingSessionImages(directoryCurrentTrainingSession + "Images\\");
	setDirectoryTrainingSessionImageAnnotations(directoryTrainingSessionImages + "Image Annotations\\");
	setDirectoryTrainingSessionInputDataFiles(directoryCurrentTrainingSession + "Input Data Files\\");
	setDirectoryTrainingSessionTargetOutputDataFiles(directoryCurrentTrainingSession + "Target Output Data Files\\");
	setDirectoryTrainingSessionNeuralNetworkArchives(directoryCurrentTrainingSession + "Neural Network Archives\\");
	setFilePathTrainingSessionListImageAnnotations(directoryCurrentTrainingSession);
	setFilePathTrainingSessionListTrainingImages(directoryCurrentTrainingSession);
	setFilePathTrainingSessionListValidationImages(directoryCurrentTrainingSession);
	setFilePathTrainingSessionListTestImages(directoryCurrentTrainingSession);	
	setTrainingSessionListImageAnnotations();
	setTrainingSessionListTrainingImages();
	setTrainingSessionListValidationImages();
	setTrainingSessionListTestImages();
	verifyTrainingSessionImageSizesMatch();
	initializeTrainingSessionNeuralNetwork();
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
	verifyTrainingSessionImageSizesMatch();
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
	verifyTrainingSessionImagesHaveImageAnnotations();
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
}
void ImageTrainingDataset::verifyTrainingSessionImagesHaveImageAnnotations(void)
{
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
	std::vector<std::string> listValidImageAnnotationFiles;
	for (auto const& dir_entry : std::filesystem::directory_iterator(directoryTrainingSessionImages)) {
		if (std::filesystem::is_regular_file(dir_entry)) {
			std::string filePathImageAnnotations = directoryTrainingSessionImageAnnotations + dir_entry.path().stem().string() + ".txt";
			if (!std::filesystem::exists(filePathImageAnnotations)) {
				std::cout << "Error ITD.000: \n";
				abort();
			}
			listValidImageAnnotationFiles.push_back(filePathImageAnnotations);
		}
	}
	for (auto const& dir_entry : std::filesystem::directory_iterator(directoryTrainingSessionImageAnnotations)) {
		if (std::filesystem::is_regular_file(dir_entry)) {
			bool validFile = false;
			for (auto const& filePath : listValidImageAnnotationFiles) {
				if (dir_entry.path().string() == filePath) { validFile = true; }
			}
			if (!validFile) {
				std::filesystem::remove(dir_entry);
			}
		}
	}
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
}
void ImageTrainingDataset::updateFileTrainingSessionListImageAnnotations(void) // Review later because of list of image annotations
{
	if (debugConsole) { std::cout << "Update ITD.000: Updating the training session annotation list text file.\n"; }
	for (auto const& dir_entry : std::filesystem::directory_iterator(directoryTrainingSessionImageAnnotations)) {
		if (std::filesystem::is_regular_file(dir_entry) && std::filesystem::path(dir_entry).extension() == ".txt") {
			std::fstream fileImageAnnotations;
			fileImageAnnotations.open(dir_entry, std::ios::in);
			if (!fileImageAnnotations.is_open()) {
				if (debugConsole) { std::cout << "Error ITD.000: Failed to open a training session image annotations text file.\n"; }
				abort();
			}
			std::string annotation;
			do {
				std::getline(fileImageAnnotations, annotation);
				if (!annotation.empty() && !fileImageAnnotations.fail()) {
					std::fstream fileAnnotationList;
					fileAnnotationList.open(filePathTrainingSessionListImageAnnotations, std::ios::in);
					if (!fileAnnotationList.is_open()) {
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
						fileAnnotationList.open(filePathTrainingSessionListImageAnnotations, std::ios::app);
						if (!fileAnnotationList.is_open()) {
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
	setTrainingSessionListImageAnnotations();
}
void ImageTrainingDataset::updateTrainingSessionInputDataFiles(void)
{
	if (debugConsole) { std::cout << "Update ITD.000: Creating the training session input data text files.\n"; }
	for (auto const& dir_entry : std::filesystem::directory_iterator(directoryTrainingSessionImages)) {
		if (std::filesystem::is_regular_file(dir_entry)) {
			// Add Comment Here Later
			cv::Mat inputImage = cv::imread(dir_entry.path().string());
			if (inputImage.empty()) {
				if (debugConsole) { std::cout << "Error ITD.000: \n"; }
				abort();
			}
			// Convert image to grayscale
			cv::Mat grayscaleImage;
			cv::cvtColor(inputImage, grayscaleImage, cv::COLOR_BGR2GRAY);
			// Add Comment Here Later
			std::string filePathInputData = directoryTrainingSessionInputDataFiles + std::filesystem::path(dir_entry).stem().string() + ".IDF.txt";
			std::fstream fileInputData;
			fileInputData.open(filePathInputData, std::ios::out);
			if (!fileInputData.is_open()) {
				if (debugConsole) { std::cout << "Error ITD.000: \n"; }
				abort();
			}
			// Iterate through pixels and write to file
			for (unsigned row = 0; row < grayscaleImage.rows; row++) {
				for (unsigned col = 0; col < grayscaleImage.cols; col++) {
					if (row == 0) {
						if (col != 0) {
							fileInputData << std::endl;
						}
					}
					else {
						fileInputData << std::endl;
					}
					double pixelValue = grayscaleImage.at<uchar>(row, col); // Access grayscale pixel value
					fileInputData << pixelValue / 255.0;
				}
			}
			fileInputData.close();
		}
	}
	if (debugConsole) { std::cout << "Update ITD.000: Finished creating the training session input data text files.\n"; }
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
			if (!fileImageAnnotations.is_open()) {
				if (debugConsole) { std::cout << "Error ITD.000: \n"; }
				abort();
			}
			do {
				std::getline(fileImageAnnotations, line);
				if (!line.empty() && !fileImageAnnotations.fail()) {
					imageAnnotations.push_back(line);
				}
			} while (!line.empty() && !fileImageAnnotations.fail());
			fileImageAnnotations.close();
			// Comment Here
			std::string filePathTargetOutputData = directoryTrainingSessionTargetOutputDataFiles + std::filesystem::path(dir_entry).stem().string() + ".TODF.txt";
			std::fstream fileTargetOutputData;
			fileTargetOutputData.open(filePathTargetOutputData, std::ios::out);
			if (!fileTargetOutputData.is_open()) {
				if (debugConsole) { std::cout << "Error ITD.000: \n"; }
				abort();
			}
			for (std::string annotationFromList : trainingSessionListImageAnnotations) {
				bool found = false;
				for (std::string annotationFromImage : imageAnnotations) {
					if (annotationFromList == annotationFromImage) { found = true; }
				}
				found ? fileTargetOutputData << "1\n" : fileTargetOutputData << "0\n";
			}
			// Comment Here (add bias node on output layer...)
			fileTargetOutputData << "1\n";
			fileTargetOutputData.close();
		}
	}
	if (debugConsole) { std::cout << "Update ITD.000: Finished creating the training session target output data text files.\n"; }
}
void ImageTrainingDataset::updateTrainingSessionTrainingValidationAndTestFiles(const double& percentTrainingData, const double& percentValidationData, const double& percentTestData)
{
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
	if (percentTrainingData + percentValidationData + percentTestData != 1.0) {
		if (debugConsole) { std::cout << "Error ITD.000: \n"; }
		abort();
	}
	unsigned numberOfImages = 0;
	for (auto const& dir_entry : std::filesystem::directory_iterator(directoryTrainingSessionInputDataFiles)) {
		if (std::filesystem::is_regular_file(dir_entry)) { numberOfImages++; }		
	}
	if (numberOfImages < 3) {
		if (debugConsole) { std::cout << "Error ITD.000: \n"; }
		abort();
	}
	// Add Comment Here Later
	unsigned numberOfTrainingImages = std::round(numberOfImages * percentTrainingData);
	if (numberOfTrainingImages == 0) { numberOfTrainingImages = 1; }
	unsigned numberOfValidationImages = std::round(numberOfImages * percentValidationData);
	if (numberOfValidationImages == 0) { numberOfValidationImages = 1; }
	unsigned numberOfTestImages = std::round(numberOfImages * percentTestData);
	if (numberOfTestImages == 0) { numberOfTestImages = 1; }
	unsigned sumOfImages = numberOfTrainingImages + numberOfValidationImages + numberOfTestImages;
	// Add Comment Here Later
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
	// Add Comment Here Later
	trainingSessionListTrainingImages.clear();
	trainingSessionListValidationImages.clear();
	trainingSessionListTestImages.clear();
	std::fstream fileTrainingImagesList;
	std::fstream fileValidationImagesList;
	std::fstream fileTestImagesList;
	fileTrainingImagesList.open(filePathTrainingSessionListTrainingImages, std::ios::out);
	fileValidationImagesList.open(filePathTrainingSessionListValidationImages, std::ios::out);
	fileTestImagesList.open(filePathTrainingSessionListTestImages, std::ios::out);
	if (!fileTrainingImagesList.is_open() || !fileValidationImagesList.is_open() || !fileTestImagesList.is_open()) {
		if (debugConsole) { std::cout << "Error ITD.000: \n"; }
		abort();
	}
	// Add Comment Here Later
	for (auto const& dir_entry : std::filesystem::directory_iterator(directoryTrainingSessionInputDataFiles)) {
		if (std::filesystem::is_regular_file(dir_entry)) {
			if (trainingSessionListTrainingImages.size() < numberOfTrainingImages) {
				trainingSessionListTrainingImages.push_back(dir_entry.path().string());
				fileTrainingImagesList << dir_entry.path().string() << std::endl;
			}
			else if (trainingSessionListValidationImages.size() < numberOfValidationImages) {
				trainingSessionListValidationImages.push_back(dir_entry.path().string());
				fileValidationImagesList << dir_entry.path().string() << std::endl;
			}
			else if (trainingSessionListTestImages.size() < numberOfTestImages) {
				trainingSessionListTestImages.push_back(dir_entry.path().string());
				fileTestImagesList << dir_entry.path().string() << std::endl;
			}
			else {
				if (debugConsole) { std::cout << "Error ITD.000: \n"; }
				abort();
			}
		}
	}
	// Add Comment Here Later
	fileTrainingImagesList.close();
	fileValidationImagesList.close();
	fileTestImagesList.close();
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
}
void ImageTrainingDataset::updateTrainingSession(const double& percentTrainingData, const double& percentValidationData, const double& percentTestData)
{
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
	updateDirectoryTrainingSessionImages();	
	updateDirectoryTrainingSessionImageAnnotations();
	updateFileTrainingSessionListImageAnnotations();
	updateTrainingSessionInputDataFiles();
	updateTrainingSessionTargetOutputDataFiles();
	updateTrainingSessionTrainingValidationAndTestFiles(percentTrainingData, percentValidationData, percentTestData);
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
}

void ImageTrainingDataset::updateFileTrainingSessionDatasetConfiguration(void)
{
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
	std::fstream fileDatasetConfiguration;
	fileDatasetConfiguration.open(filePathTrainingSessionDatasetConfiguration, std::ios::out);
	if (!fileDatasetConfiguration.is_open()) {
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
	fileDatasetConfiguration << "Directory Training Session Input Data Files:\n" << directoryTrainingSessionInputDataFiles << "\n\n";
	fileDatasetConfiguration << "Directory Training Session Target Output Data Files:\n" << directoryTrainingSessionTargetOutputDataFiles << "\n\n";
	fileDatasetConfiguration << "Directory Training Session Neural Network Archives:\n" << directoryTrainingSessionNeuralNetworkArchives << "\n\n";
	fileDatasetConfiguration << "File Path Training Session Annotation List:\n" << filePathTrainingSessionListImageAnnotations << "\n\n";
	fileDatasetConfiguration << "File Path Training Session Training Images List:\n" << filePathTrainingSessionListTrainingImages << "\n\n";
	fileDatasetConfiguration << "File Path Training Session Validation Images List:\n" << filePathTrainingSessionListValidationImages << "\n\n";
	fileDatasetConfiguration << "File Path Training Session Test Images List:\n" << filePathTrainingSessionListTestImages << "\n\n";
	fileDatasetConfiguration << "Training Session Width Of Input Images: " << trainingSessionImageWidth << std::endl;
	fileDatasetConfiguration << "Training Session Height Of Input Images: " << trainingSessionImageHeight << std::endl;
	fileDatasetConfiguration << "Training Session Number Of Image Annotations: " << trainingSessionListImageAnnotations.size() << "\n\n";
	fileDatasetConfiguration << "Training Session Neural Network Input Layer Size: " << trainingSessionImageWidth * trainingSessionImageHeight + 1 << std::endl;
	fileDatasetConfiguration << "Training Session Neural Network Output Layer Size: " << trainingSessionListImageAnnotations.size() + 1 << std::endl;
	fileDatasetConfiguration << "Training Session Neural Network Eta: " << trainingSessionNeuralNetworkEta << std::endl;
	fileDatasetConfiguration << "Training Session Neural Network Alpha: " << trainingSessionNeuralNetworkAlpha << std::endl;
	fileDatasetConfiguration << "Training Session Neural Network Topology: ";
	for (unsigned topologyIndex = 0; topologyIndex < trainingSessionNeuralNetworkTopology.size(); topologyIndex++) {
		fileDatasetConfiguration << trainingSessionNeuralNetworkTopology[topologyIndex] + 1 << (topologyIndex != trainingSessionNeuralNetworkTopology.size() - 1 ? "," : "\n");
	}
	fileDatasetConfiguration.close();
	if (debugConsole) { std::cout << "Update ITD.000: \n"; }
}

void ImageTrainingDataset::newNeuralNetwork(const double& eta, const double& alpha, const std::vector<unsigned>& hiddenLayersTopology)
{
	if (debugConsole) { std::cout << "Update ITD.000: Creating a new neural network for training.\n"; }	
	setTrainingSessionNeuralNetworkEta(eta);
	setTrainingSessionNeuralNetworkAlpha(alpha);
	setTrainingSessionNeuralNetworkTopology(hiddenLayersTopology);	
	updateFileTrainingSessionDatasetConfiguration();
	trainingSessionNeuralNetwork.createNewNeuralNetwork(trainingSessionNeuralNetworkTopology, trainingSessionNeuralNetworkEta, trainingSessionNeuralNetworkAlpha);
	if (debugConsole) { std::cout << "Update ITD.000: Finished creating a new neural network for training.\n"; }
}
void ImageTrainingDataset::existingNeuralNetwork(const std::string& fileStem)
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

void ImageTrainingDataset::runTrainingImages(void)
{
	if (debugConsole) { std::cout << "Update ITD.000: \n"; };	
	for (auto const& filePathInputDataFile : trainingSessionListTrainingImages) {
		// Add Comment Here Later
		std::fstream fileInputData;
		fileInputData.open(filePathInputDataFile, std::ios::in);
		if (!fileInputData.is_open()) {
			if (debugConsole) { std::cout << "Error ITD.000: \n"; };
			abort();
		}
		std::vector<double> dataInput;
		std::string data;
		do {
			std::getline(fileInputData, data);
			if (!fileInputData.fail() && !data.empty()) {
				dataInput.push_back(std::stod(data));
			}
		} while (!fileInputData.fail() && !data.empty());
		fileInputData.close();
		// Feed Forward
		trainingSessionNeuralNetwork.feedForward(dataInput);
		// Add Comment Here Later			
		std::string filePathTargetOutputDataFile = std::filesystem::path(filePathInputDataFile).stem().string();
		filePathTargetOutputDataFile = directoryTrainingSessionTargetOutputDataFiles + filePathTargetOutputDataFile.substr(0, filePathTargetOutputDataFile.length() - 4) + ".TODF.txt";
		if (!std::filesystem::exists(filePathTargetOutputDataFile)) {
			if (debugConsole) { std::cout << "Error ITD.000: \n"; };
			abort();
		}
		std::fstream fileTargetOutputData;
		fileTargetOutputData.open(filePathTargetOutputDataFile, std::ios::in);
		if (!fileTargetOutputData.is_open()) {
			if (debugConsole) { std::cout << "Error ITD.000: \n"; };
			abort();
		}
		std::vector<double> dataTargetOutput;
		do {
			std::getline(fileTargetOutputData, data);
			if (!fileTargetOutputData.fail() && !data.empty()) {
				dataTargetOutput.push_back(std::stod(data));
			}
		} while (!fileTargetOutputData.fail() && !data.empty());
		fileTargetOutputData.close();
		// Back Propagation
		trainingSessionNeuralNetwork.backPropagation(dataTargetOutput);
	}
	if (debugConsole) { std::cout << "Update ITD.000: \n"; };
}
void ImageTrainingDataset::runValidationImages(void)
{

}
void ImageTrainingDataset::runTestImages(void)
{
	
}
