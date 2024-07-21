# Neural-Networks
This project started simply by wanting to learn a little bit about neural networks. After watching a [video](https://vimeo.com/19569529) by Dave Miller about coding a neural network in C++, I decided to give it a try. You can also read his article about the video [here](https://millermattson.com/dave/?p=54). The neural network that I use is the same as the one Dave uses in his video. I added a way to archive the neural network and a way to load an archived neural network. I also focused on writing a class for managing an image dataset with different training sessions. I think my code is easy to follow but if you have questions feel free to ask.

- [Visual Studio](https://visualstudio.microsoft.com/downloads/)
- [OpenCV](https://opencv.org/releases/)

Nicolai Nielson has a good video on YouTube that explains installing and using OpenCV in Visual Studio.
- [OpenCV C++ and Microsoft Visual Studio: A Complete Tutorial on Installation and Usage for Beginners](https://www.youtube.com/watch?v=trXs2r6xSnI)

Everything seems to work fine so far and I plan on adding more to it as I have time. A good starting point for anyone needing one.

# Getting Started
To get started I have created a small example image dataset that you can use to see how everything works. Once downloaded it should look like this:

![Image 1](https://github.com/user-attachments/assets/a9a261df-79bd-4dea-a63f-5ca728bb58c2)

When you open the Images directory it should look like this:

![Image 2](https://github.com/user-attachments/assets/231ef392-87e4-4d2a-82e1-8703b4242103)

This is where you will have the main images of your dataset and in the Image Annotations directory each image will have a text file with that image's annotations. The file stem (everything before .txt) of the image annotations text file should match the file stem of the image.

You will need to modify the path to the image dataset in the Image Training Dataset.cpp to where you have downloaded the dataset to on your computer.

When you use the dataset for the first time it will create an additional directory called Training Sessions. This is to help keep what you are working on seperate from the main dataset. It should look like this:

![Image 3](https://github.com/user-attachments/assets/29a63da5-8bd1-4551-a07c-85e2fcd0db2c)

In the Training Sessions directory it will create the default training session directory for you and it should look like this:

![Image 4](https://github.com/user-attachments/assets/ac96aa71-2ccb-4975-8bca-aa0a9e1bac2d)

Each training session directory will have the following layout:

![Image 5](https://github.com/user-attachments/assets/f820e238-a435-4db5-88dc-698a7c430660)

# Training Sessions
The main reason I wanted to have different training sessions is because over time the main image dataset could change. Images could be added or removed from the dataset and the annotations for the images could change. Each training session creates a copy of the images and their annotation text files when you update the training session. I recommend only calling the method to update a training session once. This creates a snapshot of the dataset for the training session.

When the training session is updated it starts by copying all of the images from the dataset's main Images directory and then verifies that all of the images are the same size. It is important that all of the training session's images are the same size because when the neural network is created later on, the size of the images is used to determine the size of the input layer of the neural network. In the example image dataset the images are 72 x 72. This means that the input layer of the neural network will be 72 x 72 = 5184 inputs plus one bias input giving us 5185 as the size of the neural network's input layer.

Next, all of the image annotation text files are copied from the dataset's main Image Annotations directory and then verifies that all of the images have image annotation text files. Each image annotations text file is then read, and each new annotation is added to the training sessions List of Image Annotations text file. The List of Image Annotations is what determines the size of the output layer of the neural network. In the example image dataset there are 15 different image annotations. This means that the output layer of the neural network will have 15 outputs plus one bias output giving us 16 as the size of the neural network's output layer. The bias output on this layer is later ignored.

Next, the Input Data Files are created. Each image will have its own input data text file. The stem of each input data text file will be the same as the stem of it's respective image with ".IDF" appended to it before the text file extension. Each image is converted to grayscale and then each pixel value is divided by 255. This allows each pixel value to be a value from zero to one. Each line of an input data file represents a different pixel from left to right and top to bottom of it's respective image. 

Next, the Target Output Data Files are created. Each image will have its own target output data text file. The stem of each target output data text file will be the same as the stem of it's respective image with ".TODF" appended to it before the text file extension. Each line of a target output data file represents an image annotation from the List of Image Annotations text file plus one bias output. A zero means that the corresponding image annotation line does not apply and a one means that the corresponding image annotation line does apply. The following image shows the 15 different image annotations we discussed earlier for the example image dataset and one of the target output data files that has 16 lines (15 plus the one bias output that is later ignored):

![Image 6](https://github.com/user-attachments/assets/5d0ceb4a-3a6e-4bd6-a727-01d8d3b43775)

