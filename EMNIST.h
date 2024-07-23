#pragma once

// C++ Libraries
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// OpenCV Libraries
#include <opencv2/opencv.hpp>

void extractEMNISTImagesAndLabels(const std::string& filePathSourceImages, const std::string& filePathSourceLabels, const std::string& directoryImages, const std::string& directoryImageAnnotations, const unsigned& indexStarting)
{
	std::cout << "Update: Converting the EMNIST files.\n";

	// Open The EMNIST Files
	std::fstream fileEMNISTImages;
	std::fstream fileEMNISTLabels;
	fileEMNISTImages.open(filePathSourceImages, std::ios::binary | std::ios::in);
	fileEMNISTLabels.open(filePathSourceLabels, std::ios::binary | std::ios::in);
	if (!fileEMNISTImages.is_open() || !fileEMNISTLabels.is_open()) {
		std::cout << "Error: Failed to open the EMNIST files.\n";
		abort();
	}

	// Add Comment Here Later
	char magicNumber[4];
	char numberOfImages[4];
	char numberOfRows[4];
	char numberOfColumns[4];
	char numberOfLabels[4];

	fileEMNISTImages.read(magicNumber, 4);
	fileEMNISTImages.read(numberOfImages, 4);
	fileEMNISTImages.read(numberOfRows, 4);
	fileEMNISTImages.read(numberOfColumns, 4);
	fileEMNISTLabels.read(magicNumber, 4);
	fileEMNISTLabels.read(numberOfLabels, 4);

	// Add Comment Here Later
	int numberImages = (static_cast<unsigned char>(numberOfImages[0]) << 24) | (static_cast<unsigned char>(numberOfImages[1]) << 16) | (static_cast<unsigned char>(numberOfImages[2]) << 8) | (static_cast<unsigned char>(numberOfImages[3]));
	int numberRows = (static_cast<unsigned char>(numberOfRows[0]) << 24) | (static_cast<unsigned char>(numberOfRows[1]) << 16) | (static_cast<unsigned char>(numberOfRows[2]) << 8) | (static_cast<unsigned char>(numberOfRows[3]));
	int numberCols = (static_cast<unsigned char>(numberOfColumns[0]) << 24) | (static_cast<unsigned char>(numberOfColumns[1]) << 16) | (static_cast<unsigned char>(numberOfColumns[2]) << 8) | (static_cast<unsigned char>(numberOfColumns[3]));
	int numberLabels = (static_cast<unsigned char>(numberOfLabels[0]) << 24) | (static_cast<unsigned char>(numberOfLabels[1]) << 16) | (static_cast<unsigned char>(numberOfLabels[2]) << 8) | (static_cast<unsigned char>(numberOfLabels[3]));

	// Add Comment Here Later
	if (numberImages != numberLabels) {
		std::cout << "Error: Number of images and number of labels do not match.\n";
		abort();
	}

	// For Each Image
	for (int i = 0; i < numberImages; i++) {

		// Get The EMNIST Image
		std::vector<unsigned char> image(numberRows * numberCols);
		fileEMNISTImages.read((char*)(image.data()), numberRows * numberCols);

		// Create A Temporary OpenCV Image Of The EMNIST Image
		cv::Mat temporaryImage = cv::Mat::zeros(cv::Size(28, 28), CV_8UC1);
		int pixelIndex = 0;
		for (int row = 0; row < numberRows; row++) {
			for (int col = 0; col < numberCols; col++) {
				temporaryImage.at<uchar>(cv::Point(col, row)) = (int)image[pixelIndex];
				pixelIndex++;
			}
		}

		// Save The Image File
		std::string filePathImage = directoryImages.find_last_of("\\") == directoryImages.length() - 1 ? directoryImages : directoryImages + "\\";
		filePathImage += "EMNIST Image " + std::to_string(i + indexStarting) + ".jpg";
		cv::imwrite(filePathImage, temporaryImage);

		// Create An Image Annotations File For The Image File
		std::string filePathImageAnnotations = directoryImageAnnotations.find_last_of("\\") == directoryImageAnnotations.length() - 1 ? directoryImageAnnotations : directoryImageAnnotations + "\\";
		filePathImageAnnotations += "EMNIST Image " + std::to_string(i + indexStarting) + ".txt";
		std::fstream fileImageAnnotations;
		fileImageAnnotations.open(filePathImageAnnotations, std::ios::out);
		if (!fileImageAnnotations.is_open()) {
			std::cout << "ERROR: Failed to create an image annotations file.\n";
			abort();
		}

		// Add Comment Here Later - Also what if its a letter not a number...
		std::vector<unsigned char> label(1);
		fileEMNISTLabels.read((char*)(label.data()), 1);
		int number = static_cast<unsigned char>(label[0]);
		fileImageAnnotations << std::to_string(number);
		fileImageAnnotations.close();
	}

	// Close The EMNIST Files
	fileEMNISTImages.close();
	fileEMNISTLabels.close();

	std::cout << "Update: Finished converting the EMNIST files.\n";
}