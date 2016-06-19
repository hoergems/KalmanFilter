#ifndef _KALMAN_FILTER_HPP_
#define _KALMAN_FILTER_HPP_

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/timer.hpp>

#include <Eigen/Dense>

namespace shared {

class KalmanFilter {
public:
	KalmanFilter();
	
	void kalmanPredict(Eigen::VectorXd &x, 
			           Eigen::VectorXd &u,
			           Eigen::MatrixXd &A,
			           Eigen::MatrixXd &B,
			           Eigen::MatrixXd &P_t,
			           Eigen::MatrixXd &V,
			           Eigen::MatrixXd &M,
			           Eigen::VectorXd &x_predicted,
			           Eigen::MatrixXd &P_predicted);
	
	void kalmanPredictPy(std::vector<double> &x,
			             std::vector<double> &u,
			             std::vector<std::vector<double>> &A,
			             std::vector<std::vector<double>> &B,
			             std::vector<std::vector<double>> &P_t,
			             std::vector<std::vector<double>> &V,
			             std::vector<std::vector<double>> &M,
			             std::vector<double> &x_predicted,
			             std::vector<std::vector<double>> &P_predicted);
	
	void kalmanUpdate(Eigen::VectorXd &x_predicted,
			          Eigen::VectorXd &z,
			          Eigen::MatrixXd &H,
			          Eigen::MatrixXd &predictedCovariance,
			          Eigen::MatrixXd &W,
			          Eigen::MatrixXd &N,
			          Eigen::VectorXd &x_estimated,
			          Eigen::MatrixXd &estimatedCovariance);
	
	void kalmanUpdatePy(std::vector<double> &x,
			            std::vector<double> &z,
			            std::vector<std::vector<double>> &H,
			            std::vector<std::vector<double>> &predictedCovariance,
			            std::vector<std::vector<double>> &W,
			            std::vector<std::vector<double>> &N,
			            std::vector<double> &x_estimated,
			            std::vector<std::vector<double>> &estimatedCovariance);
	
	void computePredictedCovariance(Eigen::MatrixXd &A,
			                        Eigen::MatrixXd &P_t,
			                        Eigen::MatrixXd &V,
			                        Eigen::MatrixXd &M,
			                        Eigen::MatrixXd &predictedCovariance);
	
	void computeKalmanGain(Eigen::MatrixXd &H, 
			               Eigen::MatrixXd &predictedCovariance,
			               Eigen::MatrixXd &W, 
			               Eigen::MatrixXd &N,
			               Eigen::MatrixXd &kalmanGain);
	
	void computeStateEstimate(Eigen::VectorXd &x_predicted,
			                  Eigen::VectorXd &z,
			                  Eigen::MatrixXd &H,
			                  Eigen::MatrixXd &kalmanGain,
			                  Eigen::VectorXd &stateEstimate);
	
	void computeEstimatedCovariance(Eigen::MatrixXd &kalmanGain,
			                        Eigen::MatrixXd &H,
			                        Eigen::MatrixXd &predictedCovariance,
			                        Eigen::MatrixXd &estimatedCovariance);
	
	void computeLGains(std::vector<Eigen::MatrixXd> &A, 
			           std::vector<Eigen::MatrixXd> &B, 
			           Eigen::MatrixXd &C, 
			           Eigen::MatrixXd &D,
			           std::vector<Eigen::MatrixXd> &gains);
	
	
};

}

#endif