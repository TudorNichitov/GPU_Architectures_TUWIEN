#include <iostream>
#include <fstream>
#include <stdio.h>
#include<stdlib.h>
/* run this program using the console pauser or add your own getch, system("pause") or input loop */

using std::cout; using std::cerr;
using std::endl; using std::string;
using std::ifstream;
using namespace std;

void Generate_Graph()
{
	// read from input file (input.txt)
	// on the 1st line we have the number of nodes
	// 2nd line number of edges
	// 3rd line weight ranges
	
	 ifstream newfile("input.txt");

	string line; 
	
	int v[3] ;
	
	
	if (newfile.is_open())  
    {
        for (int i = 0; i < 3; i++) 
        {
            newfile >> v[i];
        }
    }
     
  	int n = v[0];
 	int e = v[1];
  	int w = v[2];
    
	newfile.close(); //close the file object.

	cout<<"\n Number of nodes:"<<n;
	cout<<"\n Number of edges: "<<e;
	cout<<"\n Weight ranges: "<<w;
	//we can also take the inputs from console but I used a txt file

	int i, j, edge[e][2], count;
	i=0;

	// building connections between two random nodes

 	while(i < e)
   {
   	
   		//assign directions of edges (nodes they connect)
      edge[i][0] = rand()%n+1;
      edge[i][1] = rand()%n+1;
      
      if(edge[i][0] == edge[i][1])
         { //reassign if they don't connect diff nodes. 
		 continue;}
      else
      {
         for(j = 0; j < i; j++)
         { //reassign if they are identical to previous edges 
            if((edge[i][0] == edge[j][0] && edge[i][1] == edge[j][1]) || (edge[i][0] == edge[j][1] && edge[i][1] == edge[j][0]))
            i--;
         }
      }i++;
   }
   
   // output the graph in the txt file and also calculate and output a random weight for each edge 

std::ofstream outfile ("output.txt");

//1st line
outfile << "H" << " "<< n << " "<< e<< " 0"<<std::endl;


   cout<<"\n Generating the graph ";
   for(j = 0; j < e; j++)
   {
     
      outfile << "E"<< " "; 
      outfile <<edge[j][0]<<" ";
      outfile <<edge[j][1]<<" ";
      
      // calculate and output the random weight
      int wgt = rand() % w + 1; 
      outfile <<wgt<<std::endl;
      
   }
     
   outfile.close();
    cout<<"\n Check output.txt for the graph"; 
}
	
	


int main(int argc, char** argv) {
	
	Generate_Graph();
	
	return 0;
}
