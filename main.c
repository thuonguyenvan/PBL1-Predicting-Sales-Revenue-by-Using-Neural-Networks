#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <conio.h>
#include <string.h>

int INPUT_COUNT=4, HIDDEN_COUNT=5, OUTPUT_COUNT=1, TRAINING_SET, DATASET_SIZE, VALIDATION_SET;
double* dataset;

double sigmoid(double x){
    return 1.0/(1+exp(-x));
}

double dSigmoid(double x){
    return (x)*(1-x);
}

void getData(){
    FILE* f;
    char name[100];
    printf("Input the name of the file you want to open: ");
    scanf("%s",name);
    strcat(name,".txt");
    f=fopen(name, "r");
    if (NULL == f) {
        printf("File can't be opened.\n");
        exit(1);
    }
    else printf("File read successfully.\n\n");
    fscanf(f, "%d", &DATASET_SIZE);
    VALIDATION_SET=DATASET_SIZE/10;
    TRAINING_SET=DATASET_SIZE-VALIDATION_SET-INPUT_COUNT;
    dataset = (double*)calloc(DATASET_SIZE,sizeof(double));
    fscanf(f, "%lf", &dataset[0]);
    for (int i=1; i<DATASET_SIZE; i++){
        fscanf(f, ",%lf", &dataset[i]);
    }
    fclose(f);
}

double predict(double input[], double inputWeights[][HIDDEN_COUNT], double hiddenWeights[], double hiddenBias[], double *outputBias){
    double hiddenLayer[HIDDEN_COUNT], output=0;
    for (int i=0; i<HIDDEN_COUNT; i++) hiddenLayer[i]=0;

    for (int j=0; j<HIDDEN_COUNT; j++){
        for (int i=0; i<INPUT_COUNT; i++) hiddenLayer[j]+=inputWeights[i][j]*input[i];   // Tinh mot node cua hidden layer
        hiddenLayer[j]+=hiddenBias[j];
        hiddenLayer[j]=sigmoid(hiddenLayer[j]);     // Dua qua ham sigmoid
        output+=hiddenLayer[j]*hiddenWeights[j];
    }
    output+=(*outputBias);
    output=sigmoid(output);
    return output;
}

void learn(double input[][INPUT_COUNT], double outputReal[], double learnRate, double inputWeights[][HIDDEN_COUNT], double hiddenWeights[], double hiddenBias[], double *outputBias){
    long long loop=0;

    double inputWeightsIn[INPUT_COUNT][HIDDEN_COUNT], hiddenWeightsIn[HIDDEN_COUNT], hiddenBiasIn[HIDDEN_COUNT], outputBiasIn;
    for (int i=0; i<HIDDEN_COUNT; i++){
        for (int j=0; j<INPUT_COUNT; j++) inputWeightsIn[j][i] = inputWeights[j][i];
        hiddenWeightsIn[i] = hiddenWeights[i];
        hiddenBiasIn[i]=hiddenBias[i];
    }
    outputBiasIn=*outputBias;

    double error, hiddenError[HIDDEN_COUNT];
    int epoch=0;
    while (epoch++<100000){
        for (int t=0; t<TRAINING_SET; t++){

            double hiddenLayer[HIDDEN_COUNT], output=0;
            for (int i=0; i<HIDDEN_COUNT; i++) hiddenLayer[i]=0;

            for (int j=0; j<HIDDEN_COUNT; j++){
                for (int i=0; i<INPUT_COUNT; i++) hiddenLayer[j]+=inputWeights[i][j]*input[t][i];   // Tinh mot node cua hidden layer
                hiddenLayer[j]+=hiddenBias[j];
                hiddenLayer[j]=sigmoid(hiddenLayer[j]);     // Dua qua ham sigmoid
                output+=hiddenLayer[j]*hiddenWeights[j];
            }
            output+=(*outputBias);
            output=sigmoid(output);    // Tinh output hien tai

            error = (outputReal[t] - output)*dSigmoid(output);    // Tinh sai so

            for (int i=0; i<HIDDEN_COUNT; i++) hiddenError[i] = error*hiddenWeights[i]*dSigmoid(hiddenLayer[i]);    // Lan truyen tin hieu loi

            for (int j=0; j<HIDDEN_COUNT; j++){
                for (int i=0; i<INPUT_COUNT; i++) inputWeights[i][j] += learnRate*hiddenError[j]*input[t][i]/TRAINING_SET;   // Tinh lai weight cua input layer
                hiddenWeights[j] += learnRate*error*hiddenLayer[j]/TRAINING_SET;     // Tinh lai weight cua hidden layer
            }

            (*outputBias) += learnRate*error/TRAINING_SET;

            for (int i=0; i<HIDDEN_COUNT; i++) hiddenBias[i] += learnRate*hiddenError[i]/TRAINING_SET;
        }
        loop++;
    }
    
    printf("Input weights before:\n");
    for (int i=0; i<INPUT_COUNT; i++){
        for (int j=0; j<HIDDEN_COUNT; j++) printf("%g   ", inputWeightsIn[i][j]);
        printf("\n");
    }
    printf("\n");
    printf("Input weights after:\n");
    for (int i=0; i<INPUT_COUNT; i++){
        for (int j=0; j<HIDDEN_COUNT; j++) printf("%g   ", inputWeights[i][j]);
        printf("\n");
    }
    printf("\n");

    printf("Hidden weights before:\n");
    for (int i=0; i<HIDDEN_COUNT; i++) printf("%g   ", hiddenWeightsIn[i]);
    printf("\n\n");
    printf("Hidden weights after:\n");
    for (int i=0; i<HIDDEN_COUNT; i++) printf("%g   ", hiddenWeights[i]);
    printf("\n\n");

    printf("Hidden bias before:\n");
    for (int i=0; i<HIDDEN_COUNT; i++) printf("%g   ", hiddenBiasIn[i]);
    printf("\n\n");
    printf("Hidden bias after:\n");
    for (int i=0; i<HIDDEN_COUNT; i++) printf("%g   ", hiddenBias[i]);
    printf("\n\n");

    printf("Output bias before:\n%g\n\nOutput bias after:\n%g\n\n", outputBiasIn, *outputBias);
}

void giveRandWeights(double inputWeights[][HIDDEN_COUNT], double hiddenWeights[], double hiddenBias[], double *outputBias){
    srand(time(0));
    for (int i=0; i<INPUT_COUNT; i++){
        for (int j=0; j<HIDDEN_COUNT; j++) inputWeights[i][j] = (double)rand()/(double)RAND_MAX;
    }
    for (int i=0; i<HIDDEN_COUNT; i++){
        hiddenWeights[i] = (double)rand()/(double)RAND_MAX;
        hiddenBias[i] = (double)rand()/(double)RAND_MAX;
    }
    (*outputBias) = (double)rand()/(double)RAND_MAX;
}

void printMenu(){
    printf("~~~~/~~~/~~/~/// MENU ///~/~~/~~~/~~~~\n");
    printf("1. Train neural network.\n2. Validate neural network.\n3. Predict using neural network.\n4. Exit.\n");
    printf("~~~~/~~~/~~/~///~~~~~~///~/~~/~~~/~~~~\n");
}

int main(){
    getData();
    double inputWeights[INPUT_COUNT][HIDDEN_COUNT], hiddenWeights[HIDDEN_COUNT], hiddenBias[HIDDEN_COUNT], outputBias, learnRate=0.2,
        datamax=-INT_MAX;
    for (int i=0; i<DATASET_SIZE; i++)
        datamax=(datamax>dataset[i])?datamax:dataset[i];
    
    giveRandWeights(inputWeights,hiddenWeights,hiddenBias,&outputBias);

    int index=0;
    double input[TRAINING_SET][INPUT_COUNT], outputReal[TRAINING_SET];
    for (int i=0; i<TRAINING_SET; i++){
        for (int j=0; j<INPUT_COUNT; j++)
            input[i][j]=dataset[index++]/datamax;
        index-=3;
    }
    index=INPUT_COUNT;
    for (int i=0; i<TRAINING_SET; i++){
        outputReal[i]=dataset[index++]/datamax;
    }
    char temp;
    double ans, input0[INPUT_COUNT];
    do{
        fflush(stdin);
        printMenu();
        scanf("%c", &temp);
        switch(temp){
            case '1': learn(input,outputReal,learnRate,inputWeights,hiddenWeights,hiddenBias,&outputBias);
                break;
            case '2':
                index=TRAINING_SET;
                for (int i=1; i<=VALIDATION_SET; i++){
                    for (int i=0; i<INPUT_COUNT; i++) input0[i]=dataset[index++]/datamax;
                    ans=predict(input0,inputWeights,hiddenWeights,hiddenBias,&outputBias);
                    printf("Predicted sale: %g\n", ans);
                    printf("Actual sale: %g\n\n", dataset[index]/datamax);
                    index-=INPUT_COUNT-1;
                }
                break;
            case '3':
                index=DATASET_SIZE-INPUT_COUNT;
                for (int i=0; i<INPUT_COUNT; i++) input0[i]=dataset[index]/datamax;
                ans=predict(input0,inputWeights,hiddenWeights,hiddenBias,&outputBias);
                printf("Predicted sale of the 31st day: %g\n\n", ans);
                break;
            case '4': ;
                return 0;
            default: printf("Please choose a number from 1-4.\n");
                break;
        }
        printf("Press any button to continue.\n\n");
        getch();  
    } while (temp!='4');
}