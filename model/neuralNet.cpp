//
// Created by Pete Downey on 9/15/21.
//
#include <iostream>
#include <thread>
#include <fstream>
#include <string>
#include <fstream>
#include <vector>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream
#include <numeric>


using namespace std;

//modified  from
//https://www.gormanalysis.com/blog/reading-and-writing-csv-files-with-cpp/
vector<vector<float>> readCSV(std::string filename) {

    // Create an input filestream
    std::ifstream myFile(filename);

    // Make sure the file is open
    assert(myFile.is_open());
    assert(myFile.good());

    vector<vector<float>> data(130, vector<float>(1000, 0));

    for (int n = 0; n < 130; n++) {
        // Extract the lines of the file
        string singleLineData;
        std::getline(myFile, singleLineData);

        //why do you have to use a stringstrema
        stringstream singleLineDataStream(singleLineData);

        for (int n2 = 0; n2 < 1000; n2++) {
            string temp;
            std::string::size_type sz;

            std::getline(singleLineDataStream, temp, ',');
            data[n][n2] = stof(temp, &sz);
        }
    }

    // Close file
    myFile.close();

    return data;
}

std::tuple<std::vector<std::vector<float>>, std::vector<float>, std::vector<float>>
standerdize(vector<vector<float>> data) {

    vector<float> meanList = vector<float>();
    vector<float> rangeList = vector<float>();

    for (int n = 0; n < data.size(); n++) {

        float mean = std::reduce(data[0].begin(), data[0].end())/data[0].size();
        float ranges = *std::max_element(data[0].begin(), data[0].end()) -
                       *std::min_element(data[0].begin(), data[0].end());

        for (int n2 = 0; n2 < data.size(); n2++) {
            data[n][n2] = ((data[n][n2] - mean) / ranges);

            meanList.emplace_back(mean);
            rangeList.emplace_back(ranges);
        }
    }

    //data = np.column_stack(
    //        (data, np.array(range(len(data)))))  # indices are kept track of to match each row to its inverse

    return {data, meanList, rangeList};
}


[[noreturn]] void NeuralNet() {

// DATA ===========

    const char *fileName = "/Users/petedowney/Documents/GitHub/NNProject/data/data.csv";

    vector<vector<float>> data = readCSV(fileName);


// standardization
     auto [data2, meanList, rangeList] = standerdize(data);

     //I hate this
     data = data2;
     data2.clear();


    /*
    methods.distributionPlotAfter(data);

    X, Y = methods.splitData(data, output_count + 1);

// split data
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.3);
    train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size = 0.5);

    reshaped = lambda;

    x.reshape(x.shape[0], x.shape[1], 1);

    test_X = reshaped(test_X);
    train_X = reshaped(train_X);
    val_X = reshaped(val_X);

    test_Y, test_inverse = methods.snipY(test_Y);
    train_Y, train_inverse = methods.snipY(train_Y)
    val_Y, val_inverse = methods.snipY(val_Y);

// MODEL ========
    model = models.Sequential();

// input layer (pre-network convolution)
    model.add(layers.Conv1D(32, kernel_size = 8, strides = 1, input_shape = (None, 1), activation = 'swish',
                            padding = "causal"));
    model.add(layers.AveragePooling1D(2));

// LSTM
    model.add(layers.LSTM(48, activation = 'swish', input_shape = (None, 1), return_sequences = False));

// hidden layers
    model.add(layers.Dense(128, activation = 'swish'));
    model.add(layers.Dense(64, activation = 'linear'));

// output layer
    model.add(layers.Dense(output_count, activation = 'linear'));


    model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics =['accuracy']);

    history = model.fit(train_X, train_Y, epochs = 30, batch_size = 64, validation_data = (val_X, val_Y), verbose = 0);
*/

}

