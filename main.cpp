#include <iostream>
#include <random>
#include <math.h>
#include <ctime>
#include <fstream>

using namespace std;


// XOR Gate Learning



class MLP{
	public:
		double inlay[80][2]={{0,0},{0,1},{1,0},{1,1},{0,0},{0,1},{1,0},{1,1},{0,0},{0,1},{1,0},{1,1},{0,0},{0,1},{1,0},{1,1},{0,0},{0,1},{1,0},{1,1},{0,0},{0,1},{1,0},{1,1},{0,0},{0,1},{1,0},{1,1},{0,0},{0,1},{1,0},{1,1},{0,0},{0,1},{1,0},{1,1},{0,0},{0,1},{1,0},{1,1},{0,0},{0,1},{1,0},{1,1},{0,0},{0,1},{1,0},{1,1},{0,0},{0,1},{1,0},{1,1},{0,0},{0,1},{1,0},{1,1},{0,0},{0,1},{1,0},{1,1},{0,0},{0,1},{1,0},{1,1},{0,0},{0,1},{1,0},{1,1},{0,0},{0,1},{1,0},{1,1},{0,0},{0,1},{1,0},{1,1},{0,0},{0,1},{1,0},{1,1}};
		double hidlay[4];
		double outlay[1];
		double real_outlay[80][1]={{0},{1},{1},{0},{0},{1},{1},{0},{0},{1},{1},{0},{0},{1},{1},{0},{0},{1},{1},{0},{0},{1},{1},{0},{0},{1},{1},{0},{0},{1},{1},{0},{0},{1},{1},{0},{0},{1},{1},{0},{0},{1},{1},{0},{0},{1},{1},{0},{0},{1},{1},{0},{0},{1},{1},{0},{0},{1},{1},{0},{0},{1},{1},{0},{0},{1},{1},{0},{0},{1},{1},{0},{0},{1},{1},{0},{0},{1},{1},{0}};
		
		double weights1[4][2];
		double weights2[1][4];
		
		double biases1[4] = {0,0,0,0};
		double biases2[1] = {0};
		
		int i_l_num = 2;
		int h_l_num = 4;
		int o_l_num = 1;
		
		double loss[5000];
		double accuracy[5000];
		double learning_rate_w=0.07;
		double learning_rate_b=0.07;
		
		void set_weights(){
			// Initiliatizing Weights 1
			random_device rd{};
			mt19937 seed1{rd()};
			normal_distribution<> nd1{0.5,0.1};
			for (int j=0;j<h_l_num;j++){
				for (int k=0;k<i_l_num;k++){
					weights1[j][k]= nd1(seed1);
					}
				}
			// Initiliatizing Weights 2
			mt19937 seed2{rd()};
			normal_distribution<> nd2{0.5,0.1};
			for (int i=0;i<o_l_num;i++){
				for (int j=0;j<h_l_num;j++){
					weights2[i][j]= nd2(seed2);
					}
				}
			
			}
		
		int treshold(double x){
			if (x>0.5){
				return 1;}
			else if (x<=0.5){
				return 0;
				}			
			
			return 2.0;}
		
		double linear_comb(double array1[], double array2[],int m){
			double sumof = 0;
			for (int q=0; q<m; q++){
				sumof += array1[q]*array2[q];
				}
			return sumof;
			}
			
		
		double sigmoid(double lcomb){
			return (1/(1+exp(-1*(lcomb))));
			}
			
		double d_sigmoid(double x){
			return (sigmoid(x)*(1-sigmoid(x)));
			}
			
		double lossf(int snum){
			double loss;
			for (int i=0; i<o_l_num;i++){
				loss += pow((real_outlay[snum][i]-outlay[i]),2);
				}
			return (double)loss/1.0;
			}
			
		double d_lossf(int snum, int neuron){
			double loss;
			loss = -2*(real_outlay[snum][neuron]-outlay[neuron]);
			return loss;
			}
			
		void backpropagation_biases_2(int snum){
			for (int i=0; i<o_l_num; i++){
				for (int j=0; j<h_l_num;j++){
					double E_to_a = d_lossf(snum,i);
					double weights2_x[h_l_num];
					for (int p=0; p<h_l_num;p++){weights2_x[p] = weights2[i][p];}
					double a_to_z = d_sigmoid(linear_comb(hidlay,weights2_x,h_l_num)+biases2[i]);
					double z_to_b = 1;
					double d_ratio_2 = E_to_a*a_to_z*z_to_b;
					biases2[i] = biases2[i] - (learning_rate_b)*d_ratio_2;
					}}
			}
		void backpropagation_biases_1(int snum){
			double i_all;
			for (int j=0; j<h_l_num; j++){
				for (int k=0; k<i_l_num;k++){
					for (int i=0; i<o_l_num;i++){
						double i_all__E_to_a = d_lossf(snum,i);
						double weights2_x[h_l_num];
						for (int p=0; p<h_l_num;p++){weights2_x[p] = weights2[i][p];}
						double i_all__a_to_z = d_sigmoid(linear_comb(hidlay,weights2_x,h_l_num)+biases2[i]);
						double i_all__z_to_a_prev = weights2[i][j];
						i_all += i_all__E_to_a*i_all__a_to_z*i_all__z_to_a_prev;
					}
					double weights1_x[i_l_num];
					for (int p=0; p<i_l_num;p++){weights1_x[p] = weights1[j][p];}
					double inarray[i_l_num];
					for (int p=0;p<i_l_num;p++){inarray[p] = inlay[snum][p];}
					double a_prev_to_z = d_sigmoid(linear_comb(inarray,weights1_x,i_l_num)+biases1[j]);
					double z_to_b = 1;
					double d_ratio_1 = i_all*a_prev_to_z*z_to_b;
					i_all=0;
					biases1[j] = biases1[j] - (learning_rate_b)*d_ratio_1;
						
						
						}}
			}
			
		void backpropagation_weights_2_3(int snum){
			for (int i=0; i<o_l_num; i++){
				for (int j=0; j<h_l_num; j++){
					double E_to_a = d_lossf(snum,i);
					double weights2_x[h_l_num];
					for (int p=0; p<h_l_num;p++){weights2_x[p] = weights2[i][p];}
					double a_to_z = d_sigmoid(linear_comb(hidlay,weights2_x,h_l_num)+biases2[i]);
					double z_to_w = hidlay[j];
					double d_ratio_23 = E_to_a*a_to_z*z_to_w;
					weights2[i][j] = weights2[i][j] - learning_rate_w*d_ratio_23;			
					
				}}
			}
		void backpropagation_weights_1_2(int snum){
			double i_all=0;
			for (int j=0; j<h_l_num;j++){
				for (int k=0; k<i_l_num;k++){
					for (int i=0; i<o_l_num; i++){
						double i_all__E_to_a = d_lossf(snum,i);
						double weights2_x[h_l_num];
						for (int p=0; p<h_l_num;p++){weights2_x[p] = weights2[i][p];}
						double i_all__a_to_z = d_sigmoid(linear_comb(hidlay,weights2_x,h_l_num)+biases2[i]);
						double i_all__z_to_w = weights2[i][j];
						i_all += i_all__E_to_a*i_all__a_to_z*i_all__z_to_w;
						}
					double weights1_x[i_l_num];
					for (int p=0; p<i_l_num;p++){weights1_x[p] = weights1[j][p];}
					double inarray[i_l_num];
					for (int p=0;p<i_l_num;p++){inarray[p] = inlay[snum][p];}
					double E_to_z = i_all*(d_sigmoid(linear_comb(inarray,weights1_x,i_l_num))+biases1[j]);
					i_all=0;
					double d_ratio_12 = E_to_z*(inlay[snum][k]);
					weights1[j][k] = weights1[j][k] - learning_rate_w*d_ratio_12; 
					
					
					}}
			}
		
		void show_loss(int i){
			ofstream f("/home/ismailko/Documents/Projects/DL/MyMLP/c++mlp/results_xor.py");
			f << "Loss_list = [";
			for (int a=0;  a<i-1; a++)
				f << loss[a] <<", ";
			f << "]";
			f << "\n";
			f << "Accuracy_list = [";
			for (int a=0;  a<i-1; a++)
				f << accuracy[a] <<", ";
			f << "]";
			f.close();
		
		}
		
		void train(int epochnum){
			int epoch=1;
			double accuracy_now = 0;
			clock_t startTime = clock();
			double duration;
			for (epoch=1; epoch<epochnum+1; epoch++){
				
				double loss_gec = 0;
				
				for (int sample=0; sample<80;sample++){
					
					
					//Forward Propagation between layer 1 and 2
					double inarray[i_l_num];
					
					for (int p=0;p<i_l_num;p++){inarray[p] = inlay[sample][p];}
					
					for (int j=0; j<h_l_num;j++){
						double weights1_x[i_l_num];
						for (int p=0;p<i_l_num;p++){weights1_x[p] = weights1[j][p];}
						double z1 = linear_comb(inarray,weights1_x,i_l_num)+biases1[j];
						double a1 = sigmoid(z1);
						hidlay[j] = a1;
						}
					
					//Forward Propagation between layer 2 and 3	
					for (int i=0; i< o_l_num; i++){
						double weights2_x[h_l_num];
						for (int p=0;p<h_l_num;p++){weights2_x[p] = weights2[i][p];}
						double z2 = linear_comb(hidlay,weights2_x,h_l_num)+biases2[i];
						double a2 = sigmoid(z2);
						outlay[i] = a2;
						int bin = treshold(a2);
						if 	(real_outlay[sample][0] == bin){
							accuracy_now += (double)1/(20*4);
							cout << accuracy_now << endl;
							}		
						}				
						
					//Back Propagation between layer 2 and 3
					backpropagation_biases_2(sample);
					backpropagation_weights_2_3(sample);
					
					//Back Propagation between layer 1 and 2
					backpropagation_biases_1(sample);
					backpropagation_weights_1_2(sample);
					
					
					
					loss_gec += lossf(sample);
					cout << "Epoch "<<epoch<<"/"<<epochnum<<endl;
					cout << sample << "/" << "80 --- Loss: " << lossf(sample)<<endl;
					cout << '\r';
					}
					
				loss[epoch-1] = loss_gec/80;
				accuracy[epoch-1] = accuracy_now/((double)epoch);
				loss_gec = 0;	
				}
				
				clock_t endTime = clock();
				duration = (endTime-startTime)/(double) CLOCKS_PER_SEC;
				show_loss(epoch);
				cout << endl;
				cout << "Accuracy: " << accuracy[epochnum-1]<<endl;
				cout << "Loss: " << loss[epochnum-1]<<endl;
				cout << "Duration: " << duration <<" sec";
				}
				
				
	};

int main(){
	MLP mlp;
	mlp.set_weights();
	mlp.train(5000);

return 0;
}

