# Load libraries
import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn as sk
import plotly.graph_objects as go
import pyhrv.time_domain as td
import future
import sys
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from scipy.signal import find_peaks
from scipy import signal
from scipy.ndimage.filters import uniform_filter1d
from future.moves import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from pyhrv.tools import heart_rate


def summarizeData(dataset):
    # dimensions of the data
    print(dataset.shape)
    # print first 20 rows of data
    print(dataset.head(20))
    # statistics of the data e.g. mean, std, min, quartiles
    print(dataset.describe())
    # shows the distribution of the data in each class
    #print(dataset.groupby('Stressed').size())


def createFeatures(array):
    print("Begin Creating Features")
    #Filtering GSR
    #Set up GSR filters - first phasic signal
    soslp = signal.butter(1, 5, btype='lowpass', output='sos', fs=100)
    soshp = signal.butter(1, 0.05, btype='highpass', output='sos', fs=100)
    # GSRPhasic=signal.savgol_filter(array[:,1],53,3)
    #Apply lowpass filter then high pass filter
    GSRPhasic = signal.sosfiltfilt(soslp, array[:,1])
    GSRPhasic = signal.sosfiltfilt(soshp, GSRPhasic)
    # difference of the skin conductance using the value t-1 and t, first diff = 0
    GSR_P_Diff = np.diff(GSRPhasic, prepend = GSRPhasic[0])
    # GSR_P_Diff = signal.savgol_filter(array[:,1],53,3,1)

    #Now generating tonic signal
    GSRTonic = uniform_filter1d(array[:,1], size=500)
    GSR_T_Diff = np.diff(GSRTonic, prepend=GSRTonic[0])

    # Heart Rate Calcs
    #set up heart rate filters
    soslp = signal.butter(4, 3.5, btype='lowpass', output='sos', fs=100)
    soshp = signal.butter(4, 0.833, btype='highpass', output='sos', fs=100)

    BVPFiltered = signal.sosfiltfilt(soslp, array[:,2])
    BVPFiltered = signal.sosfiltfilt(soshp, BVPFiltered)

    # Calculate BPM
    print("BPM Calculation")
    scaler = MinMaxScaler()
    peakTemp = scaler.fit_transform(BVPFiltered.reshape(-1, 1))
    peaks = find_peaks(peakTemp.reshape(-1), width=5, distance=40, prominence=0.02)[0]
    peaksMS = array[peaks,0]
    BPMArray = heart_rate(rpeaks=peaksMS)
    BPMAvg4 = []
    temp = 0
    x = 0
    for index in range(len(array[:,0])):
        for peak in range(x,len(peaks)):
            if index < peaks[peak]:
                x = peak
                temp = (BPMArray[peak - 1] + BPMArray[peak - 2] + BPMArray[peak - 3] + BPMArray[peak - 4]) / 4
                if peak < 4:
                    temp = (BPMArray[0] + BPMArray[1] + BPMArray[2] + BPMArray[3]) / 4
                break
        BPMAvg4.append(temp)

    #HRV Calcs
    print("HRV Calculation")
    trunc = []
    HRV = []
    y = 0
    trigger = False
    for index in range(len(array[:,0])):
        trunc.clear()
        for peakIndex in range(y,len(peaksMS)):
            if peaksMS[peakIndex] > array[index,0] + 15:
                break
            if peaksMS[peakIndex] > array[index,0] - 15:
                trunc.append(peaksMS[peakIndex])
                if not trigger:
                    y = peakIndex
                    trigger = True
        trigger = False
        #Careful - I believe this line below will error if trunc is empty for some indexes - there should be heart beats in 30 second interval though.
        HRV.append(td.rmssd(rpeaks=trunc)['rmssd'])

    array = np.c_[array, BPMAvg4]
    array = np.c_[array, GSRPhasic]
    array = np.c_[array, GSRTonic]
    array = np.c_[array, GSR_P_Diff]
    array = np.c_[array, GSR_T_Diff]
    array = np.c_[array, BVPFiltered]
    array = np.c_[array, HRV]
    array = np.c_[array, BPMAvg4]

    #Normalize Features used for ML
    data = array[:,5:]
    trans = joblib.load('scaler.sav')
    data = trans.transform(data)
    # trans = MinMaxScaler()
    # data = trans.fit_transform(data)
    # joblib.dump(trans, 'scaler.sav')
    array[:,5:] = data
    #Rearrange columns if stressed column was present initially
    i = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 3]
    array = array[:, i]

    #output to CSV
    print("Saving to FeatureOutput.csv")
    np.set_printoptions(suppress=True)
    np.savetxt("FeatureOutput.csv", array, fmt='%f', header="Time, GSR Ohms, BVP Raw, BPM Avg 4, GSR Phasic, GSR Tonic, GSR Phasic Diff, GSR Tonic Diff, BVP Filtered, HRV, BPM Normalized, Stressed", delimiter=",")
    print("Done Creating Features")
    return array


def compareAlgorithms(array):
    print("Begin Comparison")
    X = array[:, 4:11]
    y = array[:, 11]
    # Split-out validation and training dataset
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)
    # creating list of models to compare
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))
    models.append(('DTree', DecisionTreeClassifier(max_depth=1)))
    models.append(('NB', GaussianNB()))
    models.append(('LSVM', LinearSVC(loss='hinge', max_iter=50000)))
    models.append(('ANN', MLPClassifier(random_state=1, hidden_layer_sizes = (10,), max_iter=1000)))

    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:

    #Loading models trained on different dataset
        filename = '%s_Model.sav' % (name)
        # loaded_model = joblib.load(filename)
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    # If wanting to train models, uncomment below and it'll export the models you train for future use
    #     model.fit(X_train, Y_train)
    #     joblib.dump(model, filename)

    print("Comparison Complete - Plotting results")
    # Compare Algorithms
    plt.pyplot.boxplot(results, labels=names)
    plt.pyplot.title('Algorithm Comparison')
    plt.pyplot.ion()
    plt.pyplot.show()


def predict(array):
    # Split-out validation dataset
    print("Predicting")
    X = array[:, 4:11]
    Y = array[:, 11]
    # X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)
    # Make predictions on validation dataset
    model = joblib.load('NB_Model.sav')
    predictions = model.predict(X)
    array = np.c_[array, predictions]

    np.set_printoptions(suppress=True)
    np.savetxt("Predictions.csv", array, fmt='%f',
               header="Time, GSR Ohms, BVP Raw, BPM Avg 4, GSR Phasic, GSR Tonic, GSR Phasic Diff, GSR Tonic Diff, BVP Filtered, HRV, BPM Normalized, Stressed, Predicted",
               delimiter=",")
    # Evaluate predictions

    print(f1_score(Y, predictions, average='weighted'))
    print(confusion_matrix(Y, predictions))
    print(classification_report(Y, predictions))
    return array

def visualizeData(array):
    print("Begin Visualization")
    fig = go.Figure()
    fig.update_layout(title_text="Stress Watch Classification - BPM", font=dict(size=20))

    #adding traces (mapping dataset with name and axis on figure)
    # fig.add_trace(go.Scatter(x=array[:, 0], y=array[:, 2], name="BVP", yaxis="y"))
    fig.add_trace(go.Scattergl(x=array[:, 0], y=array[:, 3], name="Avg BPM", yaxis="y1"))
    # fig.add_trace(go.Scattergl(x=array[:, 0], y=array[:, 8], name="BVP Filtered", yaxis="y2"))
    # fig.add_trace(go.Scattergl(x=array[:, 0], y=array[:, 1], name="GSR Ohms", yaxis="y3"))
    # fig.add_trace(go.Scattergl(x=array[:, 0], y=array[:, 9], name="HRV (30s)", yaxis="y3"))
    # fig.add_trace(go.Scattergl(x=array[:, 0], y=array[:, 4], name="GSR Phasic", yaxis="y4"))
    # fig.add_trace(go.Scattergl(x=array[:, 0], y=array[:, 5], name="GSR Tonic", yaxis="y5"))
    # fig.add_trace(go.Scattergl(x=array[:, 0], y=array[:, 4], name="Filt & Diff Skin Conductance", yaxis="y5"))

    fig.update_traces(
        hovertemplate="%{y:.2f}",
        line={"width": 2},
        mode="lines",
        showlegend=False,
    )

    #Drawing properties for each y axis
    fig.update_layout(
        xaxis=dict(
            # rangeslider=dict(autorange=True,
            #                         visible=False,
            #                         borderwidth=1
            #                         ),
                   mirror=False,
                   ticks="outside",
                   showline=True,
                   title="Time, s",
                   rangemode="tozero"
                   ),

        yaxis=dict(anchor="x",
                   autorange=True,
                   domain=[.1, 0.8],
                   linecolor="#673ab7",
                   showline=True,
                   side="right",
                   tickfont={"color": "#673ab7"},
                   tickmode="auto",
                   ticks="outside",
                   title="Avg BPM",
                   titlefont={"color": "#673ab7"},
                   type="linear",
                   zeroline=False,
                   mirror=True,
                   showgrid=True, gridwidth=1
                   ),
        # yaxis2=dict(anchor="x",
        #             autorange=True,
        #             domain=[0.4, 0.8],
        #             linecolor="#E91E63",
        #             showline=True,
        #             side="right",
        #             tickfont={"color": "#E91E63"},
        #             tickmode="auto",
        #             ticks="outside",
        #             title="BVP Filtered",
        #             titlefont={"color": "#E91E63"},
        #             type="linear",
        #             zeroline=False,
        #             mirror=True,
        #             ),
        # yaxis3=dict(anchor="x",
        #             autorange=True,
        #             domain=[0.38, 0.57],
        #             linecolor="#795548",
        #             showline=True,
        #             side="right",
        #             tickfont={"color": "#795548"},
        #             tickmode="auto",
        #             ticks="outside",
        #             title="GSR Ohms",
        #             titlefont={"color": "#795548"},
        #             type="linear",
        #             zeroline=False,
        #             mirror=True,
        #             ),
        # yaxis4=dict(anchor="x",
        #             range=[0, 1],
        #             domain=[0.57, .76],
        #             linecolor="#607d8b",
        #             showline=True,
        #             side="right",
        #             tickfont={"color": "#607d8b"},
        #             tickmode="auto",
        #             ticks="outside",
        #             title="GSR Phasic",
        #             titlefont={"color": "#607d8b"},
        #             type="linear",
        #             zeroline=False,
        #             mirror=True,
        #             ),
        # yaxis5=dict(anchor="x",
        #             range=[0, 1],
        #             domain=[0.76, .95],
        #             linecolor="#673ab7",
        #             showline=True,
        #             side="right",
        #             tickfont={"color": "#673ab7"},
        #             tickmode="auto",
        #             ticks="outside",
        #             title="GSR Tonic",
        #             titlefont={"color": "#673ab7"},
        #             type="linear",
        #             zeroline=False,
        #             mirror=True,
        #             ),

    )
    #adding legend and template to figure
    fig.update_layout(
        dragmode="zoom",
        hovermode="x",
        legend=dict(traceorder="reversed", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=1000,
        template="simple_white",
        margin=dict(
            t=100,
            b=100
        )
    )

    # #Adding Axis lines
    fig.add_shape(type="line",
                  xref="paper",
                  yref="paper",
                  x0=0,
                  y0=1,
                  x1=1,
                  y1=1,
                  line=dict(color="black", width=2)
                  )
    fig.add_shape(type="line",
                  xref="paper",
                  yref="paper",
                  x0=0,
                  y0=0.8,
                  x1=1,
                  y1=0.8,
                  line=dict(color="black",width=2)
                  )
    # fig.add_shape(type="line",
    #               xref="paper",
    #               yref="paper",
    #               x0=0,
    #               y0=0.57,
    #               x1=1,
    #               y1=0.57,
    #               line=dict(color="black", width=2)
    #               )
    fig.add_shape(type="rect",
                  xref="paper",
                  yref="paper",
                  x0=0.628,
                  y0=1.015,
                  x1=.643,
                  y1=1.045,
                  line=dict(color="rgba(0, 0, 255, .5)", width=3, ),
                  fillcolor='rgba(0, 0, 255, 0.5)',
                  )
    fig.add_shape(type="rect",
                  xref="paper",
                  yref="paper",
                  x0=0.86,
                  y0=1.015,
                  x1=0.875,
                  y1=1.045,
                  line=dict(color="rgba(255, 0, 0, 1)", width=3, ),
                  fillcolor='rgba(255, 0, 0, 0.5)',
                  )
    fig.add_shape(type="line",
                  xref="paper",
                  yref="paper",
                  x0=0,
                  y0=.8,
                  x1=0,
                  y1=1,
                  line=dict(color="black", width=2)
                  )
    fig.add_shape(type="line",
                  xref="paper",
                  yref="paper",
                  x0=1,
                  y0=.8,
                  x1=1,
                  y1=1,
                  line=dict(color="black", width=2)
                  )
    fig.add_annotation(x=.85,
                       y=1.048,
                       xref="paper",
                       yref="paper",
                       text="Machine Learning Predicted Stress",
                       showarrow=False,
                       font=dict(size=20),
                       )

    fig.add_annotation(x=.98,
                       y=1.048,
                       xref="paper",
                       yref="paper",
                       text="Subjective Stress",
                       showarrow=False,
                       font=dict(size=20),
                       )
    # x=.968 for Induced Stress

    #Drawing stressed areas
    stressed = False
    temp = []
    start = 0
    for x in range(0, len(array[:, 11])):
        if array[x, 11] == 1 and not stressed:
            stressed = True
            start = array[x, 0]
        if array[x, 11] == 0 and stressed:
            stressed = False
            temp.append((start, array[x-1, 0]))
    if stressed:
        temp.append((start, array[-1, 0]))
    for pair in temp:
        fig.add_shape(type="rect",
                      xref="x",
                      yref="paper",
                      x0=pair[0],
                      y0=0.83,
                      x1=pair[1],
                      y1=0.89,
                      line=dict(color="rgba(255, 0, 0, .5)", width=3, ),
                      fillcolor='rgba(255, 0, 0, 0.5)',
                      layer="below"
                      )
        # Adding Stressed annotation
        # fig.add_annotation(x=(pair[0]+pair[1])/2,
        #                    y=1,
        #                    xref="x",
        #                    yref="paper",
        #                    text="Stressed",
        #                    align="center",
        #                    showarrow=False,
        #                    )

    # #Drawing Predicted areas
    predicted = False
    temp = []
    start = 0
    for x in range(0, len(array[:, 12])):
        if array[x, 12] == 1 and not predicted:
            predicted = True
            start = array[x, 0]
        if array[x, 12] == 0 and predicted:
            predicted = False
            temp.append((start, array[x - 1, 0]))
    if predicted:
        temp.append((start, array[-1, 0]))
    for pair in temp:
        fig.add_shape(type="rect",
                      xref="x",
                      yref="paper",
                      x0=pair[0],
                      y0=.91,
                      x1=pair[1],
                      y1=0.97,
                      line=dict(color="rgba(0, 0, 255, .5)", width=3, ),
                      fillcolor='rgba(0, 0, 255, 0.5)',
                      layer="below"
                      )
        # Adding predicted annotation
        # fig.add_annotation(x=(pair[0] + pair[1]) / 2,
        #                    y=1,
        #                    xref="x",
        #                    yref="paper",
        #                    text="Predicted",
        #                    align="center",
        #                    showarrow=False,
        #                    )
    ###########Temp BPM visualization - VERY SLOW#################################
    # peaks = find_peaks(array[:,8], width=5, distance=40, prominence=0.08)[0]
    # for peak in peaks:
    #     fig.add_shape(type="line",
    #                   xref="x",
    #                   yref="y2",
    #                   x0=array[peak,0],
    #                   y0=np.min(array[:,8]),
    #                   x1=array[peak,0],
    #                   y1=np.max(array[:,8]),
    #                   line=dict(
    #                       color="LightSeaGreen",
    #                       width=3,
    #                   )
    #               )
    #################################################################
    fig.show()



    stressArray = []
    for index in array[:,12]:
        if index == 0:
            stressArray.append("Non-Stressed")
        if index == 1:
            stressArray.append("Stressed")

    fig1 = go.Figure()
    fig1.update_layout(title_text="Percentage of Time Stressed", font=dict(size=20))
    fig1.add_trace(go.Histogram(histfunc="count",
                                histnorm='percent',
                                y=array[:,12],
                                x=stressArray,
                                name='Machine Learning Prediction',
                                hoverinfo='skip',
                                )
                   )
    fig1.update_traces(hovertemplate="%{y:0f}", showlegend=True)

    fig1.update_layout(
        template="simple_white",
    )
    fig1.update_yaxes(ticksuffix='%', showgrid=True, gridcolor='black', gridwidth=2)
    fig1.show()



    print("Done Visualization")



###############                         Main program starts here                              ######################
windowMain = tk.Tk()
windowMain.title("Smart Watch Stress Classification")
windowMain.geometry("500x100")
lbl = tk.Label(windowMain, text="Input CSV")
lbl.grid(row=0, column=0)

def clicked():
    filePath = askopenfilename(title="Select CSV File", filetypes=(("CSV files", "*.csv"),))
    txt['state'] = tk.NORMAL
    if len(txt.get()) is not 0:
        txt.delete(0,tk.END)
    txt.insert(0,filePath)
    txt['state'] = tk.DISABLED

txt = tk.Entry(windowMain, width=50, state=tk.DISABLED)
txt.grid(row=0, column=1)
btn = tk.Button(windowMain, text="Browse...", command=clicked)
btn.grid(row=0, column=2)

def close(window):
    if window is windowMain:
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            sys.exit()
        return
    window.withdraw()

tk.Button(windowMain, text="Close Program", command=lambda:close(windowMain)).grid(row=2, column=0)

def run():
    if len(txt.get()) is 0:
        messagebox.showinfo("No File!", message="Please Select CSV File")
        return

    dataset = pd.read_csv(txt.get())

    # convert dataframe to numpy array
    array = dataset.to_numpy()

    # summarizeData(dataset)
    array = createFeatures(array)
    # compareAlgorithms(array)
    array = predict(array)
    visualizeData(array)

btnStart = tk.Button(windowMain, text="Run Program", command=run).grid(row=1, column=0)

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        sys.exit()

windowMain.protocol("WM_DELETE_WINDOW", on_closing)
windowMain.mainloop()