# circPred
circPred, focused on distinguish circular RNAs from protein coding transcripts using support vector machine. Firstly we extracted features, including sequence length, GC content, frequencies of GT and AG, A-to-I density and Alu repeats binding scores from transcripts. Secondly, 128 motif features are selected by convolutional neural network. Then, we use SVM to predict circRNAs.

Requirements:
1. Python3
2. Keras
3. Numpy
4. sklearn
5. joblib

How to use the tool, the command as follows:
python circPred.py --model_dir models/ --ref_dir reference/ --input_file data/test_data --output_file data/pred_result.txt

--model-dir: The directory to load the trained models for prediction
--ref-dir: The directory to load the AtoI file and Alu file
--input_file: JSON input file for predicting data
--output_file: The output file used to store the prediction label and probability of input data

Input file data format:
Each line in input file should match the following JSON format:
{
“id”: transcript id,
“chro”: ‘chr1’ or...
“strand”: ‘+’ or ‘-’
“up_stream”: [up_strat, up_end, up_seq],
“down_stream”: [down_start, down_end, down_seq]
}
NOTICE: as for “id”, we should have unique name for the transcript.

Output file:
transcript-id	prediction-label	prediction-probability
transcript1	0	0.1983548
transcript2	1	0.8192133

