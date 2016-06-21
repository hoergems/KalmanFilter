#ifndef _KALMAN_FILTER_HPP_
#define _KALMAN_FILTER_HPP_
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
			           Eigen::MatrixXd &P_predicted) {
		x_predicted = A * x + B * u;	
		computePredictedCovariance(A, P_t, V, M, P_predicted);
	}
	
	
	void kalmanUpdate(Eigen::VectorXd &x_predicted,
			          Eigen::VectorXd &z,
			          Eigen::MatrixXd &H,
			          Eigen::MatrixXd &predictedCovariance,
			          Eigen::MatrixXd &W,
			          Eigen::MatrixXd &N,
			          Eigen::VectorXd &x_estimated,
			          Eigen::MatrixXd &estimatedCovariance) {
		const int size = x_predicted.rows();
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
	
	void computeStateEstimate(Eigen::VectorXd &x_predicted,
			                  Eigen::VectorXd &z,
			                  Eigen::MatrixXd &H,
			                  Eigen::MatrixXd &kalmanGain,
			                  Eigen::VectorXd &stateEstimate) {
		stateEstimate = x_predicted + kalmanGain * (z - H * x_predicted);
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
	
	
};

}

#endif