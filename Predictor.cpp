//#include "stdafx.h"
#include "Predictor.h"

tensorflow::DataType Convert2DT(string type)
{
	if (type == "string")
		return DT_STRING;
	else
		return DT_STRING;
}

void split2vec(string str, char delimeter, vector<string> &strVec) //Could be Overloaded
{
	stringstream ss(str);
	string buf;
	while (getline(ss, buf, delimeter))
		strVec.push_back(buf);
	//cout << strVec.size() << endl;
}

void split2vec(string str, char delimeter, vector<int> &intVec)
{
	stringstream ss(str);
	int buf;
	while (ss >> buf)
		intVec.push_back(buf);
}

Predictor::Predictor()
{
	isInitial = false;
}

string Predictor::setup_all(string metafile, string checkpointname, string inputnode, string outputnode, string inputtype, string outputtype, string preRunOps)
{
	string errorReason("successful");
	isInitial = predictor_setup(metafile, checkpointname, errorReason);
	if (!isInitial)
		return errorReason;
	isInitial = inout_setup(inputnode, outputnode, inputtype, outputtype, preRunOps, errorReason);
	return errorReason;
}

bool Predictor::predictor_setup(string metafile, string checkpointname, string &errorReason)
{
	isInitial = true;
	Status status;
	// 01. Create session
	SessionOptions so;
	so.config.set_allow_soft_placement(true);
	session = NewSession(so);
	if (!session)
	{
		cout << "Could not create Tensorflow session." << endl;
		errorReason = "Could not create Tensorflow session.";
		return false;
	}
	cout << "1/4.Create session done!" << endl;
	// 02. Read graph
	//MetaGraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), metafile, &graph_def);
	//status = ReadBinaryProto(Env::GetFileSystemForFile(metafile), metafile, &graph_def);
	//GraphDef *graph_def_nodevice = &graph_def.graph_def();
	if (!status.ok())
	{
		cout << "Error reading graph definition from" + metafile + ": " + status.ToString() << endl;
		errorReason = "Error reading graph definition from" + metafile + ": " + status.ToString();
		return false;
	}
	/*for (int n = 0; n < graph_def.graph_def().node_size(); ++n)
		graph_def.graph_def().mutable_node(n)->clear_device();*/

	cout << "2/4.Read graph done!" << endl;
	// 03. Load graph
	status = session->Create(graph_def.graph_def());
	//status = session->Create(graph_def_nodevice);
	if (!status.ok())
	{
		cout << "Error creating graph : " + status.ToString() << endl;
		errorReason = "Error creating graph : " + status.ToString();
		return false;
	}
	cout << "3/4.Load graph done!" << endl;
	// 04. Load weight
	Tensor checkpointPathTensor(DT_STRING, TensorShape());
	checkpointPathTensor.scalar<string>()() = checkpointname;
	status = session->Run(
	{ { graph_def.saver_def().filename_tensor_name(), checkpointPathTensor }, },
	{},
	{ graph_def.saver_def().restore_op_name() },
	nullptr);
	//cout << graph_def.saver_def().restore_op_name() << endl;
	//cout << graph_def.DebugString() << endl;
	//graph_def.saver_def().rel
	if (!status.ok())
	{
		cout <<" Error loading checkpoint from " + checkpointname + ": " + status.ToString() << endl;
		errorReason = " Error loading checkpoint from " + checkpointname + ": " + status.ToString();
		return false;
	}
	cout << "4/4.Restore weight done!" << endl;
	return true;
}

bool Predictor::inout_setup(string inputnode, string outputnode, string inputtype, string outputtype, string preRunOps, string &errorReason)
{
	split2vec(inputnode, ',', inName);
	split2vec(outputnode, ',', outName);
	split2vec(inputtype, ',', inType);
	split2vec(outputtype, ',', outType);
	split2vec(preRunOps, ',', preops);
	if (inName.size() != inType.size() || outName.size() != outType.size())
	{
		isInitial = false;
		errorReason = "Name and Type not align";
		return false;
	}
	return true;
}

void Predictor::predict(vector<vector<string>> &input, vector<vector<string>> &output)
{
	//for (int i = 0; i < input.size(); i++)
	//{
	//	auto a_map = inputTensor[i].tensor<string, 1>();  //Not necessarily string, will modify later.
	//	a_map(0) = input[i];
	//}
	//
	//status = session->Run(inputTensor, { output }, {}, &answer);
	//01. Create input
	if (!isInitial)
	{
		cout << "Model is not initialized!" << endl;
	}
	vector<pair<string, Tensor>> inputTensor;
	tensorflow::TensorShape inputshape({ int(input[0].size()) });
	for (int i = 0; i < input.size(); i++)
	{
		Tensor a(Convert2DT(inType[i]), inputshape);
		auto a_map = a.tensor<string, 1>(); //Not necessarily string, will modify later.
		for (int j = 0; j < input[0].size(); j++)
			a_map(j) = input[i][j];
		inputTensor.emplace_back(std::string(inName[i]), a);
	}
	//cout << "input build done." << endl;
	vector<Tensor> answer;
	vector<Tensor> waste;
	//02. Make the prediction
	if(preops.size())
		session->Run({}, {}, preops, &waste);
	session->Run(inputTensor, outName, {}, &answer);
	//cout << "run done." << endl;
	//cout << "answer size: " << answer.size() << endl;
	//cout << outName[0] << "\t" << outName[1] << endl;
	for (int i = 0; i < outName.size(); i++){
		vector<string> ans;
		//cout << i << endl;
		if (outType[i] == "floatScalar")
		{
			auto answer_map = answer[i].tensor<float, 1>();
			for (int j = 0; j < input[0].size(); j++)
				ans.push_back(to_string(answer_map(j)));
		}
		if (outType[i] == "floatVector")
		{
			auto answer_map = answer[i].tensor<float, 2>();
			int vecSize = answer_map.size() / input[0].size();
			for(int j = 0; j < input[0].size(); j++)
			{
				string vec;
				for (int k = 0; k < vecSize; k++)
				{
					if (vec.size())
						vec += ",";
					vec += to_string(answer_map(j*vecSize + k));
				}
				ans.push_back(vec);
			}
		}
		if (outType[i] == "stringVector")
		{
			auto answer_map = answer[i].tensor<string, 2>();
			int vecSize = answer_map.size() / input[0].size();
			for (int j = 0; j < input[0].size(); j++)
			{
				string vec;
				for (int k = 0; k < vecSize; k++)
				{
					if (vec.size())
						vec += ",";
					vec += answer_map(j*vecSize + k);
				}
				ans.push_back(vec);
			}

		}
		output.push_back(ans);
	}
	return;
}
