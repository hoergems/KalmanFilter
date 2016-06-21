#ifndef _KALMAN_FILTER_HPP_
#define _KALMAN_FILTER_HPP_
#include <Eigen/Dense>
#include <robot_environment/robot_environment.hpp>
#include "utils.hpp"

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
			           Eigen::MatrixXd &P_predicted) {
		x_predicted = A * x + B * u;	
		computePredictedCovariance(A, P_t, V, M, P_predicted);
	}
	
	
	void kalmanUpdate(std::vector<double> &x_predicted,
			          std::vector<double> &z,
			          Eigen::MatrixXd &H,
			          Eigen::MatrixXd &predictedCovariance,
			          Eigen::MatrixXd &W,
			          Eigen::MatrixXd &N,
			          std::vector<double> &x_estimated,
			          Eigen::MatrixXd &estimatedCovariance) {
		x_estimated.clear();
		const int size = x_predicted.size();
		Eigen::MatrixXd kalmanGain(size, size);
		computeKalmanGain(H, predictedCovariance, W, N, kalmanGain);	
		computeStateEstimate(x_predicted, z, H, kalmanGain, x_estimated);
		computeEstimatedCovariance(kalmanGain, H, predictedCovariance, estimatedCovariance);
	}
	
	void computePredictedCovariance(Eigen::MatrixXd &A,
			                        Eigen::MatrixXd &P_t,
			                        Eigen::MatrixXd &V,
			                        Eigen::MatrixXd &M,
			                        Eigen::MatrixXd &predictedCovariance) {
		predictedCovariance = A * (P_t * A.transpose()) + (V * M) * V.transpose();
	}
	
	void computeKalmanGain(Eigen::MatrixXd &H, 
			               Eigen::MatrixXd &predictedCovariance,
			               Eigen::MatrixXd &W, 
			               Eigen::MatrixXd &N,
			               Eigen::MatrixXd &kalmanGain) {
		Eigen::MatrixXd H_transpose = H.transpose();
		Eigen::MatrixXd res_temp = H * (predictedCovariance * H_transpose);
		Eigen::MatrixXd res_temp_inv = res_temp.inverse();
		kalmanGain = predictedCovariance * (H_transpose * (H * (predictedCovariance * H_transpose) + W * (N * W.transpose())).inverse());
	}
	
	void computeStateEstimate(std::vector<double> &x_predicted,
			                  std::vector<double> &z,
			                  Eigen::MatrixXd &H,
			                  Eigen::MatrixXd &kalmanGain,
			                  std::vector<double> &stateEstimate) {
		Eigen::VectorXd x_predicted_e = utils::toEigenVec(x_predicted);
		Eigen::VectorXd z_e = utils::toEigenVec(z);	
		Eigen::VectorXd stateEstimate_e = x_predicted_e + kalmanGain * (z_e - H * x_predicted_e);
		stateEstimate = utils::toStdVec(stateEstimate_e);
	}
	
	void computeEstimatedCovariance(Eigen::MatrixXd &kalmanGain,
			                        Eigen::MatrixXd &H,
			                        Eigen::MatrixXd &predictedCovariance,
			                        Eigen::MatrixXd &estimatedCovariance) {
		Eigen::MatrixXd KtH = kalmanGain * H;
		const int size = KtH.rows();
	    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(size, size);
		estimatedCovariance = (I - KtH) * predictedCovariance;
	}
	
	void computeLGains(std::vector<Eigen::MatrixXd> &A, 
			           std::vector<Eigen::MatrixXd> &B, 
			           Eigen::MatrixXd &C, 
			           Eigen::MatrixXd &D,
			           unsigned int &horizon,
			           std::vector<Eigen::MatrixXd> &gains) {
		Eigen::MatrixXd S(C);
		std::vector<Eigen::MatrixXd> As(A);
		std::vector<Eigen::MatrixXd> Bs(B);
		std::reverse(As.begin(), As.end());
		std::reverse(Bs.begin(), Bs.end());
		for (size_t i = 0; i < horizon; i++) {		
			Eigen::MatrixXd A_tr = As[i].transpose();
			Eigen::MatrixXd B_tr = Bs[i].transpose();
			Eigen::MatrixXd L = -(B_tr * S * Bs[i] + D).inverse() * B_tr * S * As[i];
			gains.push_back(L);
			S = C + A_tr * S * As[i] + A_tr * S * Bs[i] * L;
		}
			
		std::reverse(gains.begin(), gains.end());
	}
	
	void ekfPredictState(std::shared_ptr<shared::RobotEnvironment> &env,
			             std::vector<double> &x_estimated,
			             std::vector<double> &u,
			             double &control_duration,
			             double &simulation_step_size,
			             Eigen::MatrixXd &A,
			             Eigen::MatrixXd &B,
			             Eigen::MatrixXd &V,
			             Eigen::MatrixXd &M,
			             Eigen::MatrixXd &P_t,
			             std::vector<double> &x_predicted,
			             Eigen::MatrixXd &P_predicted) {
		x_predicted.clear();
		std::vector<double> control_error;
		for (size_t i = 0; i < env->getRobot()->getControlSpaceDimension(); i++) {
			control_error.push_back(0.0);
		}
		
		env->getRobot()->propagateState(x_estimated, 
				                        u, 
				                        control_error, 
				                        control_duration,
				                        simulation_step_size,
				                        x_predicted);		
		
		Eigen::VectorXd x = utils::toEigenVec(x_predicted);
		Eigen::VectorXd u_e = utils::toEigenVec(u);	
		
		Eigen::VectorXd x_predicted_e;		
		kalmanPredict(x, u_e, A, B, P_t, V, M, x_predicted_e, P_predicted);
		x_predicted = utils::toStdVec(x_predicted_e);		
		
	}
	
};

}

#endif