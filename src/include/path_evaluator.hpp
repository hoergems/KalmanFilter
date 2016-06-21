#ifndef _PATH_EVALUATOR_HPP_
#define _PATH_EVALUATOR_HPP_
#include <iostream>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include <boost/thread.hpp>
#include <boost/timer.hpp>
#include <queue>
#include <robot_environment/robot_environment.hpp>
#include "kalman_filter.hpp"
#include <robot_environment/Obstacle.hpp>
#include "fcl/collision_object.h"
#include <path_planner/dynamic_path_planner.hpp>
#include <path_planner/Options.hpp>

namespace shared {

struct PathEvaluationResult {
	PathEvaluationResult() = default;
	
	//~PathEvaluationResult();
	
	PathEvaluationResult(PathEvaluationResult& res) {
		xs = res.xs;
		us = res.us;
		zs = res.zs;
		control_durations = res.control_durations;		
		path_objective = res.path_objective;
	}
	
	PathEvaluationResult& operator=(PathEvaluationResult &res) {
		xs = res.xs;
		us = res.us;
		zs = res.zs;
		control_durations = res.control_durations;
		path_objective = res.path_objective;
	}
	
	std::vector<std::vector<double>> xs;
	std::vector<std::vector<double>> us;
	std::vector<std::vector<double>> zs;
	
	std::vector<double> control_durations;
	
	double path_objective;
};

std::shared_ptr<shared::DynamicPathPlanner> makeDynamicPathPlanner(std::shared_ptr<shared::RobotEnvironment> &robot_environment);

class PathEvaluator {
public:
	PathEvaluator();	
	
	void setRobotEnvironment(std::shared_ptr<shared::RobotEnvironment> &robot_environment);
	
	void setDynamicPathPlanner(std::shared_ptr<shared::DynamicPathPlanner> &dynamic_path_planner); 
	
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
	
	void setRewardModel(double &step_penalty, 
			            double &illegal_move_penalty, 
			            double &terminal_reward, 
			            double &discount_factor);
	
	double setNumSamples(unsigned int &num_samples);
	
	void planAndEvaluatePaths(const std::vector<double> &start_state,			                 
			                  Eigen::MatrixXd &P_t,
			                  unsigned int &current_step,
			                  double &timeout, 
			                  unsigned &num_threads,
							  std::shared_ptr<shared::PathPlannerOptions> &path_planner_options);
	
	void eval_thread(std::shared_ptr<std::queue<std::shared_ptr<shared::PathEvaluationResult>>> &queue,
			         const std::vector<double> &start_state,
			         Eigen::MatrixXd &P_t,
			         unsigned int &current_step,
			         double &planning_timeout,
					 std::shared_ptr<shared::PathPlannerOptions> &path_planner_options);
	
private:
	boost::mutex mtx_;
	
	std::shared_ptr<shared::DynamicPathPlanner> dynamic_path_planner_;
	
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
	
	std::shared_ptr<shared::DynamicPathPlanner> makeDynamicPathPlanner();
	
};

}

#endif