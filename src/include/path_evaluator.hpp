#ifndef _PATH_EVALUATOR_HPP_
#define _PATH_EVALUATOR_HPP_
#include <iostream>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include <robot_environment/robot_environment.hpp>
#include "kalman_filter.hpp"
#include <robot_environment/Obstacle.hpp>
#include "fcl/collision_object.h"

namespace shared {

class PathEvaluator {
public:
	PathEvaluator();	
	
	void setRobotEnvironment(std::shared_ptr<shared::RobotEnvironment> &robot_environment);
	
	void getLinearModelMatrices(std::vector<std::vector<double>> &state_path,
			                    std::vector<std::vector<double>> &control_path,
			                    std::vector<double> &control_durations,
			                    std::vector<Eigen::MatrixXd> &As,
			                    std::vector<Eigen::MatrixXd> &Bs,
			                    std::vector<Eigen::MatrixXd> &Vs,
			                    std::vector<Eigen::MatrixXd> &Ms,
			                    std::vector<Eigen::MatrixXd> &Hs,
			                    std::vector<Eigen::MatrixXd> &Ws,
			                    std::vector<Eigen::MatrixXd> &Ns);
	
	void setC(Eigen::MatrixXd &C);
	
	void setD(Eigen::MatrixXd &D);
	
	double evaluatePath(std::vector<std::vector<double>> &state_path,
                        std::vector<std::vector<double>> &control_path,
                        std::vector<double> &control_durations,
                        Eigen::MatrixXd &P_t,
                        unsigned int &current_step);
	
	double setRewardModel(double &step_penalty, 
			              double &illegal_move_penalty, 
			              double &terminal_reward, 
			              double &discount_factor);
	
	double setNumSamples(unsigned int &num_samples);
	
private:
	std::shared_ptr<shared::RobotEnvironment> robot_environment_;
	
	std::shared_ptr<shared::KalmanFilter> kalman_filter_;
	
	Eigen::MatrixXd C_;
	Eigen::MatrixXd D_;
	
	double step_penalty_;
	double illegal_move_penalty_;
	double terminal_reward_;
	double discount_factor_;
	
	unsigned int num_samples_;
	
	double getExpectedStateReward(std::vector<double> &state, Eigen::MatrixXd &cov_state);
	
	void sampleValidStates(std::vector<double> &mean, 
			               Eigen::MatrixXd &cov, 
			               unsigned int num_samples,
			               std::vector<std::vector<double>> &samples); 
	
};

}

#endif