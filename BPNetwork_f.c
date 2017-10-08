#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>

#define sample 4
#define input_neuron 2
#define hidden_neuron 2
#define output_neuron 1

#define MAX_NUM 10000000

typedef struct Data{
  double inputdata[sample][input_neuron];
  double expectdata[sample][output_neuron];
}Data;

typedef struct Network{
  double input[input_neuron];
  double input_hidden_weight[input_neuron][hidden_neuron];
  double hidden[hidden_neuron];
  double hidden_delta[hidden_neuron];
  double hidden_output_weight[hidden_neuron][output_neuron];
  double output[output_neuron];
  double output_delta[output_neuron];
}Network;

void InitNeuralNetwork(Network *NeuralNetwork)
{
  int i,j;
  //srand(time(NULL));
  for(i=0; i<input_neuron ; i++)
    for(j=0; j<hidden_neuron ;j++)
      NeuralNetwork->input_hidden_weight[i][j]=(rand()+1.0)/(RAND_MAX+1.0);
  for(i=0; i<hidden_neuron ; i++)
    for(j=0; j<output_neuron ;j++)
      NeuralNetwork->hidden_output_weight[i][j]=(rand()+1.0)/(RAND_MAX+1.0);
}

double Activation(double x){return 1/(1+exp(-x));}

void PrintOutput(double *mat, int row, int column, int index, FILE *f)
{
  int i,j,k;
  for(i=0; i<row ;i++){
    for(j=0; j<column; j++){
      if(index)
	fprintf(f,"(%d,%d): %f\t",i,j,(mat+i)[j]);
      else
	fprintf(f,"sample%d: %f\n",i,(mat+i)[j]);
    }
  }
  fprintf(f,"\n");
}

void IterNeuralNetwrok(Network *NeuralNetwork, Data train_data, double error, double eta)
{
  int num=0,n,i,j,k;
  double temp_delta,temp_E;
  double E = 1.0;
  double print_output[sample][output_neuron] = {0};
  FILE *flog, *ferr;
  
  flog = fopen("log_0.6.txt","w");
  ferr = fopen("err_0.6.txt","w");
  fprintf(flog,"##   Initialization   ##\n");
  fprintf(flog,"input-hidden-weight : ");
  PrintOutput(NeuralNetwork->input_hidden_weight[0], input_neuron, hidden_neuron, 1, flog);
  fprintf(flog,"hidden-output-weight : ");
  PrintOutput(NeuralNetwork->hidden_output_weight[0], hidden_neuron, output_neuron, 1, flog);
  fprintf(flog,"\n##   Analysis...   ##\n");
  
  while(E >= error-0.0000001 && num < MAX_NUM){
    //E = 0;
    for(n=0; n<sample; n++){
      /******    1 FORWARD    ******/
      
      for(i=0; i<hidden_neuron; i++){
	for(j=0; j<input_neuron; j++)
	  NeuralNetwork->hidden[i] += train_data.inputdata[n][j]*NeuralNetwork->input_hidden_weight[j][i];
	NeuralNetwork->hidden[i] = Activation(NeuralNetwork->hidden[i]);	    
      }
      for(i=0; i<output_neuron; i++){
	NeuralNetwork->output[i] = 0;
	for(j=0; j<hidden_neuron; j++)
	  NeuralNetwork->output[i] += NeuralNetwork->hidden[j]*NeuralNetwork->hidden_output_weight[j][i];
	NeuralNetwork->output[i] = Activation(NeuralNetwork->output[i]);
	print_output[n][i] = NeuralNetwork->output[i];
      }
	  
      /******    2 BACKWARD   ******/
      /*   2.1 Compute deviation */
      for(i=0; i<output_neuron; i++)
	NeuralNetwork->output_delta[i] = NeuralNetwork->output[i]*(1-NeuralNetwork->output[i])*(NeuralNetwork->output[i]-train_data.expectdata[n][i]);   
      for(i=0; i<hidden_neuron; i++){
	temp_delta = 0;
	for(j=0; j<output_neuron; j++)
	  temp_delta += NeuralNetwork->hidden_output_weight[i][j]*NeuralNetwork->output_delta[j];
	NeuralNetwork->hidden_delta[i] = NeuralNetwork->hidden[i]*(1-NeuralNetwork->hidden[i])*temp_delta;
      }
      /*   2.2 Update weight   */
      for(i=0; i<hidden_neuron; i++)
	for(j=0; j<output_neuron; j++)
	  NeuralNetwork->hidden_output_weight[i][j] -= eta*NeuralNetwork->output_delta[j]*NeuralNetwork->hidden[i];
      for(i=0; i<input_neuron; i++)
	for(j=0; j<hidden_neuron; j++)
	  NeuralNetwork->input_hidden_weight[i][j] -= eta*NeuralNetwork->hidden_delta[j]*train_data.inputdata[n][i];

      /****   3 Calculate Cost Function  ****/
      
      E = 0;
      for(i=0; i< output_neuron; i++){
	E += fabs((NeuralNetwork->output[i]-train_data.expectdata[n][i]));
      }
    }
    if(num %1000 ==0)
      fprintf(ferr,"%d\t%f\n",num,E);
    //E += temp_E;
    num ++;
  }
        /****   4 Output  ****/
  fprintf(flog,"##   Final    ##\n");
  fprintf(flog,"Iteration Nums : %d \n", num);
  fprintf(flog,"input-hidden-weight : ");
  PrintOutput(NeuralNetwork->input_hidden_weight[0], input_neuron, hidden_neuron, 1, flog);
  fprintf(flog, "hidden-output-weight : ");
  PrintOutput(NeuralNetwork->hidden_output_weight[0], hidden_neuron, output_neuron, 1, flog);
  fprintf(flog,"Output : \n");
  PrintOutput(print_output[0], sample, output_neuron, 0, flog);
  fclose(flog);
  fclose(ferr);
}


int main(int argc, char **argv)
{
  Network XORNetwork;
  Data Train_data={{0,0,0,1,1,0,1,1},{0,1,1,0}};
  double learning_rate = 0.6;
  double error_cutoff = 0.008;
  InitNeuralNetwork(&XORNetwork);
  IterNeuralNetwrok(&XORNetwork, Train_data, error_cutoff, learning_rate);
  return 0;
}
