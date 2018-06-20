#include "Predictor.h"

void main()
{
	const string pathToGraph = "E:/cdssm_model_cpredict.meta";
	const string checkpointPath = "E:/cdssm_model_final-640005";
	string inputnode = "query:0,doc:0";
	string inputtype = "string,string";
	string outputnode = "Sum_2:0,l2_normalize:0";
	string outputtype = "floatScalar,floatVector";
	string preRunOps = "";
	Predictor pred;
	pred.setup_all(pathToGraph, checkpointPath, inputnode, outputnode, inputtype, outputtype, preRunOps);
	string tmp_q[3] = { "hello world", "google glass","iphone x" };
	string tmp_d[3] = { "hello world", "apple stock","iphone 7 plus" };
	vector<string> input_query;
	vector<string> input_doc;
	input_query.insert(input_query.begin(), tmp_q, tmp_q + 3);
	input_doc.insert(input_doc.begin(), tmp_d, tmp_d + 3);
	vector<vector<string>> output;
	vector<vector<string>> input;
	input.push_back(input_query);
	input.push_back(input_doc);
	pred.predict(input, output);
	for (int i = 0; i < output.size(); i++)
	{
		for (int j = 0; j < output[0].size(); j++)
			cout << output[i][j] << " ";
		cout << endl;
	}
}

