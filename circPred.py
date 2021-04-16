# -*- coding: utf-8 -*-

import os
import json
import joblib
import argparse
import numpy as np
from keras.models import Model, load_model
from sklearn.preprocessing import StandardScaler

# =============================================================================
#calculate features from sequence

def get_component(up, down):
    #calculate component features of sequence
    up_GC = (up.count('G') + up.count('C'))/len(up)*100
    down_GC = (down.count('G') + down.count('C'))/len(down)*100
    GC_density = (up_GC + down_GC)/2
    
    up_GT, down_GT = up.count('GT')/len(up)*100, down.count('GT')/len(down)*100
    GT_density = (up_GT + down_GT)/2
    
    up_AG, down_AG = up.count('AG')/len(up)*100, down.count('AG')/len(down)*100
    AG_density = (up_AG + down_AG)/2
    
    return (len(up)+len(down))/2, GC_density, GT_density, AG_density

def get_atoi(atoi, s_chr, s_str, u, d):
    #calculate the AtoI density of sequence
    u_s, u_e, u_seq = u
    d_s, d_e, d_seq = d
    u_count, d_count = 0, 0
    
    for i in atoi[(s_chr, s_str)]:
        if u_s <= i <= u_e:
            u_count += 1
        if d_s <= i <= d_e:
            d_count += 1
    
    u_density, d_density = u_count/len(u_seq), d_count/len(d_seq)
    atoi_density = (u_density + d_density)/2
    
    return atoi_density

def get_alu(alu, s_chr, u_s, u_e, d_s, d_e):
    #calculate the Alu binding scores of sequence
    u_forward, u_reverse = 0, 0
    d_forward, d_reverse = 0, 0
    
    for repeat in alu[s_chr]:
        r_s, r_e, r_id, r_str = repeat
        if u_s <= r_s and r_e <= u_e:
            if r_str == '+':
                u_forward += 1
            else:
                u_reverse += 1
        if d_s <= r_s and r_e <= d_e:
            if r_str == '+':
                d_forward += 1
            else:
                d_reverse += 1
    alu_binding_scores = (u_forward*d_reverse + u_reverse*d_forward)/2
    #alu_binding_scores = (u_forward - d_forward)*(d_reverse - u_reverse)/2
    
    return alu_binding_scores
# =============================================================================

# =============================================================================
# load data and extract features from sequences

def read_ref(ref_file):
    #load the reference file (AtoI and Alu)
    with open(ref_file, 'r') as f:
        return eval(f.read())

def get_file(sample_file):
    #load the JSON inputfile
    with open(sample_file, 'r') as f:
        for lines in f.readlines():
            line = json.loads(lines)
            yield (line['id'], line['chro'], line['strand'],
                   line['up_stream'], line['down_stream'])

def one_hot_code(u_seq, d_seq):
    #encode sequence with a fixed length by one-hot-code
    r ={'A':(1, 0, 0, 0),
        'T':(0, 1, 0, 0),
        'G':(0, 0, 1, 0),
        'C':(0, 0, 0, 1)}
    
    set_len = 50
    
    return [r[i] for i in u_seq[-set_len:] + d_seq[:set_len]]

def get_all_features(sample_file, atoi_file, alu_file):
    #extract features from sequences
    info, features, seq = [], [], []
    atoi = read_ref(atoi_file)
    alu = read_ref(alu_file)
    
    for val in get_file(sample_file):
        Id, chro, strand, u, d = val
        u_s, u_e, u_seq = u
        d_s, d_e, d_seq = d
        
        info.append(Id)
        
        seq.append(one_hot_code(u_seq, d_seq))
        
        length, GC, GT, AG = get_component(u_seq, d_seq)
        atoi_density = get_atoi(atoi, chro, strand, u, d)
        alu_score = get_alu(alu, chro, u_s, u_e, d_s, d_e)
        features.append((length, GC, GT, AG, atoi_density, alu_score))
    
    features, seq = np.array(features), np.array(seq)
    
    return features, seq, info

def get_cnn_features(seq, model_file):
    #extract 128 motif features by using CNN
    model = load_model(model_file)
    dense_layer_model = Model(inputs = model.input, 
                              outputs = model.get_layer(index = 4).output)
    cnn_data = dense_layer_model.predict(seq)
    
    return cnn_data

def get_confuse_features(features, set_len, cnn_model):
    cnn_features = get_cnn_features(set_len, cnn_model)
    
    all_features = np.hstack([features, cnn_features])
    
    return all_features
# =============================================================================

# =============================================================================
# predict circular RNAs from protein coding transcripts by using trained model

def preprocess_data(x):
    #preprocess features
    scaler = StandardScaler()
    return scaler.fit_transform(x)

def get_prediction(input_file, out_file, atoi_file, alu_file,
                   cnn_model, svm_model):
    #predict the input data
    features, seq, info = get_all_features(input_file, atoi_file, alu_file)
    
    all_features = get_confuse_features(features, seq, cnn_model)
    
    x_std = preprocess_data(all_features)
    
    cnsvm = joblib.load(svm_model)
    
    y_pred = cnsvm.predict(x_std)
    y_score = cnsvm.predict_proba(x_std)[:,1]
    
    with open(out_file, 'w') as fw:
        fw.write('ID\tpredict_label\tpredict_prob\n')
        for i in range(len(y_pred)):
            fw.write('{}\t{}\t{}\n'.format(info[i], y_pred[i], y_score[i]))
# =============================================================================

def run_circPred(parser):
    model_dir, data_dir = parser.model_dir, parser.ref_dir
    
    input_file, out_file = parser.input_file, parser.output_file
    
    cnn_model = os.path.join(model_dir, 'cnn_model.h5')
    svm_model = os.path.join(model_dir, 'svm_model.pkl')
    
    atoi_file = os.path.join(data_dir, 'AtoI.txt')
    alu_file = os.path.join(data_dir, 'Alu.txt')
    
    get_prediction(input_file, out_file, atoi_file, alu_file, 
                   cnn_model, svm_model)

def parse_arguments(parser):
    
    parser.add_argument('--model_dir', type = str, default = 'models/',
                        help = 'The directory to load the trained models for prediction')
    
    parser.add_argument('--ref_dir', type = str, default = 'reference/', 
                        help = 'The directory to load the AtoI file and Alu file')
    
    parser.add_argument('--input_file', type = str, default='data/test_data',
                        help = 'JSON input file for predicting data')
    
    parser.add_argument('--output_file', type = str, default = 'data/pred_result.txt',
                        help = 'The output file used to store the prediction label and probability of input data')
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = 'circular RNAs classification from protein_coding transcripts')
    
    args = parse_arguments(parser)
    
    run_circPred(args)