# Image_Caption_Generator

This project is end to end project based on a CONV-LSTM image to text model that creates a caption for an image.
This project is created mainly using Pytorch for modelisation and some aspects of data preprocessing, MLFlow for model monitoring, orchestration, and using Torch Serve as well as ONNX to make it deployment ready.

## Requirements

All of the requirements are available in the "requirements.txt" file

## Usage 

Run the 'data_loading.ipynb' notebook first to download and load your data, followed by the 'main.py' script to load, train, evaluate and log your model and export it's weights.

For deployment, run the 'deployment_exports.py' for model scripting, tracing or for conversion to an ONNX format, all of them are deployment ready formats

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
