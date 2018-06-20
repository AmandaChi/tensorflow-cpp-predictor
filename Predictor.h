#pragma once
#include <string>
#include <vector>
#include "matmul.h"
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <math.h>
#include <eigen/Dense>
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
//#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/strings/strcat.h"
//#include "tensorflow/core/platform/logging.h"
//#include "tensorflow/core/platform/test.h"
//#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow\contrib\microsoft\kernels\dssm_ops.h"
#include "tensorflow\contrib\microsoft\kernels\unique_ops.h"
#include "tensorflow\contrib\microsoft\math\lookup.h"
#include "tensorflow\contrib\microsoft\math\native.h"
#include "tensorflow\contrib\microsoft\math\simd.h"

using namespace std;
using namespace tensorflow;

class Predictor
{
public:
	Predictor();
	bool predictor_setup(string metafile, string checkpointname, string &errorReason);
	bool inout_setup(string inputnode, string outputnode, string inputtype, string outputtype, string preRunOps, string &errorReason);
	string setup_all(string metafile, string checkpointname, string inputnode, string outputnode, string inputtype, string outputtype, string preRunOps);
	void predict(vector<vector<string>> &input, vector<vector<string>> &output);
	bool isInitial;
private:
	Session* session;
	MetaGraphDef graph_def;
	//vector<pair<string, Tensor>> input;
	//vector<Tensor> output;
	//vector<Tensor> output;
	//vector<Tensor> inputTensor;
	vector<string> inName;
	vector<string> inType;
	vector<string> outName;
	vector<string> outType;
	vector<string> preops;
};

