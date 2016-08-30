#ifndef _KALMAN_FILTER_HPP_
#define _KALMAN_FILTER_HPP_
#include <Eigen/Dense>
#include <robot_environment/robot_environment.hpp>
#include "utils.hpp"
#include <signal.h>

namespace shared
{

class KalmanFilter
{
public:
    KalmanFilter();

    void kalmanPredict(std::vector<double>& x,
                       std::vector<double>& u,
                       Eigen::MatrixXd& A,
                       Eigen::MatrixXd& B,
                       Eigen::MatrixXd& P_t,
                       Eigen::MatrixXd& V,
                       Eigen::MatrixXd& M,
                       std::vector<double>& x_predicted,
                       Eigen::MatrixXd& P_predicted) {
        Eigen::VectorXd x_e = utils_kalman::toEigenVec(x);
        Eigen::VectorXd u_e = utils_kalman::toEigenVec(u);
        Eigen::VectorXd x_predicted_e = A * x_e + B * u_e;
        x_predicted = utils_kalman::toStdVec(x_predicted_e);
        computePredictedCovariance(A, P_t, V, M, P_predicted);
    }


    void kalmanUpdate(std::vector<double>& x_predicted,
                      std::vector<double>& z_dash,
                      Eigen::MatrixXd& H,
                      Eigen::MatrixXd& predictedCovariance,
                      Eigen::MatrixXd& W,
                      Eigen::MatrixXd& N,
                      std::vector<double>& x_estimated,
                      Eigen::MatrixXd& estimatedCovariance) {
        x_estimated.clear();
        const int size = x_predicted.size();
        Eigen::MatrixXd kalmanGain(size, size);
        computeKalmanGain(H, predictedCovariance, W, N, kalmanGain);
        computeStateEstimate(x_predicted, z_dash, H, kalmanGain, x_estimated);
        computeEstimatedCovariance(kalmanGain, H, predictedCovariance, estimatedCovariance);
    }

    void computePredictedCovariance(Eigen::MatrixXd& A,
                                    Eigen::MatrixXd& P_t,
                                    Eigen::MatrixXd& V,
                                    Eigen::MatrixXd& M,
                                    Eigen::MatrixXd& predictedCovariance) {
        predictedCovariance = A * P_t * A.transpose() + V * M * V.transpose();
    }

    void computeKalmanGain(Eigen::MatrixXd& H,
                           Eigen::MatrixXd& predictedCovariance,
                           Eigen::MatrixXd& W,
                           Eigen::MatrixXd& N,
                           Eigen::MatrixXd& kalmanGain) {
        Eigen::MatrixXd H_transpose = H.transpose();
	Eigen::MatrixXd S = H * predictedCovariance * H_transpose + W * N * W.transpose();
        kalmanGain = predictedCovariance * H_transpose * S.inverse();
    }

    void computeStateEstimate(std::vector<double>& x_predicted,
                              std::vector<double>& z_dash,
                              Eigen::MatrixXd& H,
                              Eigen::MatrixXd& kalmanGain,
                              std::vector<double>& stateEstimate) {
        Eigen::VectorXd x_predicted_e = utils_kalman::toEigenVec(x_predicted);
        Eigen::VectorXd z_dash_e = utils_kalman::toEigenVec(z_dash);	
        Eigen::VectorXd stateEstimate_e = x_predicted_e + kalmanGain * z_dash_e;
        stateEstimate = utils_kalman::toStdVec(stateEstimate_e);
    }

    void computeEstimatedCovariance(Eigen::MatrixXd& kalmanGain,
                                    Eigen::MatrixXd& H,
                                    Eigen::MatrixXd& predictedCovariance,
                                    Eigen::MatrixXd& estimatedCovariance) {
        Eigen::MatrixXd KtH = kalmanGain * H;
        const int size = KtH.rows();
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(size, size);
        estimatedCovariance = (I - KtH) * predictedCovariance;
    }

    bool computeLGains(std::vector<Eigen::MatrixXd>& A,
                       std::vector<Eigen::MatrixXd>& B,
                       Eigen::MatrixXd& C,
                       Eigen::MatrixXd& D,
                       unsigned int& horizon,
                       std::vector<Eigen::MatrixXd>& gains) {
	
        Eigen::MatrixXd S = C;
        std::vector<Eigen::MatrixXd> As = A;
        std::vector<Eigen::MatrixXd> Bs = B;
        std::reverse(As.begin(), As.end());
        std::reverse(Bs.begin(), Bs.end());
	Eigen::MatrixXd L;
	Eigen::MatrixXd A_tr;
	Eigen::MatrixXd B_tr;
	Eigen::MatrixXd S_old;
        for (size_t i = 0; i < horizon; i++) {
            A_tr = As[i].transpose();
            B_tr = Bs[i].transpose();
            L = -(B_tr * S * Bs[i] + D).inverse() * B_tr * S * As[i];
            if (std::isnan(L(0, 0))) {
                return false;
            }
            gains.push_back(L);
            S_old = S;
            S = C + A_tr * S_old * As[i] + A_tr * S_old * Bs[i] * L;
            if (std::isnan(S(0, 0))) {
                return false;
            }
        }
        
        std::reverse(gains.begin(), gains.end());
	return true;
    }

    void ekfPredictState(std::shared_ptr<shared::RobotEnvironment>& env,
                         const frapu::RobotStateSharedPtr& x_estimated,
                         const frapu::ActionSharedPtr& u,
                         double& control_duration,
                         double& simulation_step_size,
                         Eigen::MatrixXd& A,
                         Eigen::MatrixXd& V,
                         Eigen::MatrixXd& M,
                         Eigen::MatrixXd& P_t,
                         frapu::RobotStateSharedPtr& x_predicted,
                         Eigen::MatrixXd& P_predicted) {
        x_predicted = nullptr;
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
        computePredictedCovariance(A, P_t, V, M, P_predicted);
    }
};

}

#endif
