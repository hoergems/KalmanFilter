#include "include/path_evaluator.hpp"
#include <unistd.h>
#include <memory>

using std::cout;
using std::endl;

namespace shared {

//Threading functions

std::unique_ptr<shared::DynamicPathPlanner> makeDynamicPathPlanner(std::shared_ptr<shared::RobotEnvironment> &robot_environment,
																   std::shared_ptr<shared::PathPlannerOptions> &options) {
	std::shared_ptr<shared::RobotEnvironment> env = robot_environment->clone();	
	std::unique_ptr<shared::DynamicPathPlanner> dyn(new shared::DynamicPathPlanner(false));			
	dyn->setup(env, options->planning_algorithm);	
	ompl::base::GoalPtr goal_region = 
				shared::makeManipulatorGoalRegion(dyn->getSpaceInformation(),
				                                  env,
				                                  options->goal_states,
				                                  static_cast<shared::ManipulatorPathPlannerOptions *>(options.get())->ee_goal_position,
				                                  options->goal_radius);
	
	dyn->setGoal(goal_region);
	dyn->setControlSampler(options->control_sampler);
	dyn->addIntermediateStates(options->addIntermediateStates);
	dyn->setNumControlSamples(options->numControlSamples);
	dyn->setRRTGoalBias(options->RRTGoalBias);
	dyn->setMinMaxControlDuration(options->min_max_control_durations);
	
	return dyn;
}


PathEvaluator::PathEvaluator():
	robot_environment_(nullptr),
	dynamic_path_planner_(nullptr),
	kalman_filter_(),
	C_(),
	D_(),
	step_penalty_(0.0),
	illegal_move_penalty_(0.0),
	terminal_reward_(0.0),
	discount_factor_(0.0),
	num_samples_(1)
{
	
}

double PathEvaluator::setNumSamples(unsigned int &num_samples) {
	num_samples_ = num_samples;
}

void PathEvaluator::setRobotEnvironment(std::shared_ptr<shared::RobotEnvironment> &robot_environment) {
	robot_environment_ = robot_environment;
}

void PathEvaluator::setDynamicPathPlanner(std::shared_ptr<shared::DynamicPathPlanner> &dynamic_path_planner) {
	dynamic_path_planner_ = dynamic_path_planner;
}

double PathEvaluator::setRewardModel(double &step_penalty, 
			                         double &illegal_move_penalty, 
			                         double &terminal_reward, 
			                         double &discount_factor) {
	step_penalty_ = step_penalty;
	illegal_move_penalty = illegal_move_penalty_;
	terminal_reward_ = terminal_reward_;
	discount_factor_ = discount_factor_;
}

void PathEvaluator::getLinearModelMatrices(std::vector<std::vector<double>> &state_path,
			                               std::vector<std::vector<double>> &control_path,
			                               std::vector<double> &control_durations,
			                               std::vector<Eigen::MatrixXd> &As,
			                               std::vector<Eigen::MatrixXd> &Bs,
			                               std::vector<Eigen::MatrixXd> &Vs,
			                               std::vector<Eigen::MatrixXd> &Ms,
			                               std::vector<Eigen::MatrixXd> &Hs,
			                               std::vector<Eigen::MatrixXd> &Ws,
			                               std::vector<Eigen::MatrixXd> &Ns) {
	std::vector<Eigen::MatrixXd> A_B_V_H_W;
	for (size_t i = 0; i < state_path.size(); i++) {
		A_B_V_H_W.clear();
		robot_environment_->getRobot()->getLinearProcessMatrices(state_path[i], control_path[i], control_durations[i], A_B_V_H_W);
		As.push_back(A_B_V_H_W[0]);
		Bs.push_back(A_B_V_H_W[1]);
		Vs.push_back(A_B_V_H_W[2]);
		Hs.push_back(A_B_V_H_W[3]);
		Ws.push_back(A_B_V_H_W[4]);
		
		Eigen::MatrixXd M;
		robot_environment_->getRobot()->getStateCovarianceMatrix(M);
		Ms.push_back(M);
		
		Eigen::MatrixXd N;
		robot_environment_->getRobot()->getObservationCovarianceMatrix(N);
		Ns.push_back(N);
	}
	
}

void PathEvaluator::setC(Eigen::MatrixXd &C) {
	C_ = C;
}

void PathEvaluator::setD(Eigen::MatrixXd &D) {
	D_ = D;
}

double PathEvaluator::evaluatePath(std::vector<std::vector<double>> &state_path,
                                   std::vector<std::vector<double>> &control_path,
                                   std::vector<double> &control_durations,
                                   Eigen::MatrixXd &P_t,
                                   unsigned int &current_step) {
	std::vector<Eigen::MatrixXd> As;
    std::vector<Eigen::MatrixXd> Bs;
	std::vector<Eigen::MatrixXd> Vs;
	std::vector<Eigen::MatrixXd> Ms;
	std::vector<Eigen::MatrixXd> Hs;
	std::vector<Eigen::MatrixXd> Ws;
	std::vector<Eigen::MatrixXd> Ns;
	
	getLinearModelMatrices(state_path, 
			               control_path, 
			               control_durations, 
			               As,
			               Bs,
			               Vs,
			               Ms,
			               Hs,
			               Ws,
			               Ns);
	double path_reward = 0.0;
	unsigned int horizon_L = state_path.size();
	std::vector<Eigen::MatrixXd> Ls;
	kalman_filter_->computeLGains(As, Bs, C_, D_, Ls);
	Eigen::MatrixXd Q_t_up(Ms[0].rows(), Ms[0].cols() + Ns[0].cols());
	Eigen::MatrixXd Q_t_down(Ns[0].rows(), Ms[0].cols() + Ns[0].cols());
	Eigen::MatrixXd Q_t_up_right = Eigen::MatrixXd::Zero(Ms[0].rows(), Ns[0].cols());
	Eigen::MatrixXd Q_t_down_left = Eigen::MatrixXd::Zero(Ns[0].rows(), Ms[0].cols());
	Q_t_up << Ms[0], Q_t_up_right;
	Q_t_down << Q_t_down_left, Ns[0];
	Eigen::MatrixXd Q_t(Q_t_up.rows() + Q_t_down.rows(), Q_t_up.cols());
	Q_t << Q_t_up,
		   Q_t_down;
	
	Eigen::MatrixXd R_t_up_right = Eigen::MatrixXd::Zero(P_t.rows(), P_t.cols());
	Eigen::MatrixXd R_t_up(P_t.rows(), 2 * P_t.cols());
	Eigen::MatrixXd R_t_down(P_t.rows(), 2 * P_t.cols());
	R_t_up << P_t, R_t_up_right;
	R_t_down << R_t_up_right, P_t;
	Eigen::MatrixXd R_t(2 * P_t.rows(), 2 * P_t.cols());
	R_t << R_t_up,
		   R_t_down;
	
	for (size_t i = 1; i < horizon_L; i++) {
		Eigen::MatrixXd P_hat_t;		 
		kalman_filter_->computePredictedCovariance(As[i], P_t, Vs[i], Ms[i], P_hat_t);
		Eigen::MatrixXd K_t;
		kalman_filter_->computeKalmanGain(Hs[i], P_hat_t, Ws[i], Ns[i], K_t);		
		kalman_filter_->computeEstimatedCovariance(K_t, Hs[i], P_hat_t, P_t);
		
		Eigen::MatrixXd F_u_l(As[i]);		
		Eigen::MatrixXd F_u_r(As[i] * Ls[i - 1]);
		Eigen::MatrixXd F_l_l(K_t * Hs[i] * As[i]);
		Eigen::MatrixXd F_l_r(As[i] + Bs[i] * Ls[i - 1] - K_t * Hs[i] * As[i]);
		
		Eigen::MatrixXd F_t(F_u_l.rows() + F_l_l.rows(), F_u_l.cols() + F_u_r.cols());
		F_t << F_u_l, F_u_r,
			   F_l_l, F_l_r;
		
		Eigen::MatrixXd G_u_l(Vs[i]);
		Eigen::MatrixXd G_l_l(K_t * Hs[i] * Vs[i]);
		Eigen::MatrixXd G_l_r(K_t * Ws[i]);
		Eigen::MatrixXd G_u_r = MatrixXd::Zero(G_u_l.rows(), G_l_r.cols());
		Eigen::MatrixXd G_t(G_u_l.rows() + G_l_l.rows(), G_u_l.cols() + G_u_r.cols());
		G_t << G_u_l, G_u_r,
			   G_l_l, G_l_r;
		
		R_t = F_t * R_t * F_t.transpose() + G_t * Q_t * G_t.transpose();
		
		Eigen::MatrixXd Gamma_t_u_l = Eigen::MatrixXd::Identity(Ls[i - 1].rows(), Ls[i - 1].cols());
		Eigen::MatrixXd Gamma_t_u_r = Eigen::MatrixXd::Zero(Gamma_t_u_l.rows(), Ls[i - 1].cols());
		Eigen::MatrixXd Gamma_t_l_l = Eigen::MatrixXd::Zero(Ls[i - 1].rows(), Gamma_t_u_l.cols());
		Eigen::MatrixXd Gamma_t_l_r(Ls[i - 1]);
		Eigen::MatrixXd Gamma_t(Gamma_t_u_l.rows() + Gamma_t_l_l.rows(), Gamma_t_u_l.cols() + Gamma_t_u_r.cols());
		Gamma_t << Gamma_t_u_l, Gamma_t_u_r,
				   Gamma_t_l_l, Gamma_t_l_r;
		
		Eigen::MatrixXd Cov(Gamma_t * R_t * Gamma_t.transpose());
		Eigen::MatrixXd cov_state(Cov.block(0, 0, P_t.rows(), P_t.cols()));
		double expected_state_reward = getExpectedStateReward(state_path[i], cov_state);
		path_reward += std::pow(discount_factor_, current_step + i) * expected_state_reward;
	}	
	
	return path_reward;
}

double PathEvaluator::getExpectedStateReward(std::vector<double> &state, Eigen::MatrixXd &cov_state) {
	double expected_state_reward = 0.0;
	std::vector<std::vector<double>> state_samples;
	sampleValidStates(state, cov_state, num_samples_, state_samples);
	bool collides = false;
	for (size_t i = 0; i < num_samples_; i++) {
		// Check for collision
		std::vector<std::shared_ptr<fcl::CollisionObject>> collision_objects;
		robot_environment_->getRobot()->createRobotCollisionObjects(state_samples[i], collision_objects);
		std::vector<std::shared_ptr<shared::Obstacle>> obstacles;
		robot_environment_->getObstacles(obstacles);
		collides = false;
		for (size_t j = 0; j < obstacles.size(); i++) { 
			if (!obstacles[j]->getTerrain()->isTraversable()) {
				if (obstacles[j]->in_collision(collision_objects)) {
					expected_state_reward -= illegal_move_penalty_;
					collides = true;
					break;
				}
			}
		}
		
		if (!collides) {
			if (robot_environment_->getRobot()->isTerminal(state_samples[i])) {
				expected_state_reward += terminal_reward_;
			}
			
			else {
				expected_state_reward -= step_penalty_;
			}
		}
		
	}
	return expected_state_reward / float(num_samples_);
}

void PathEvaluator::sampleValidStates(std::vector<double> &mean, 
			                          Eigen::MatrixXd &cov, 
			                          unsigned int num_samples,
			                          std::vector<std::vector<double>> &samples) {
	samples.clear();	
	Eigen::MatrixXd mean_matr(mean.size(), 1);
	for (size_t i = 0; i < mean.size(); i++) {
		mean_matr(i) = mean[i];
	}
	
	std::shared_ptr<shared::Robot> robot = robot_environment_->getRobot();	
	std::shared_ptr<shared::EigenMultivariateNormal<double>> distr = robot_environment_->createDistribution(mean_matr, cov);
	for (size_t i = 0; i < num_samples; i++) {
		std::vector<double> sample;
		Eigen::MatrixXd sample_m(mean.size(), 1);
		distr->nextSample(sample_m);
		for (size_t j = 0; j < mean.size(); j++) {
			sample.push_back(sample_m(j));
		}
		
		if (robot->constraintsEnforced()) {
			robot->enforceConstraints(sample);
		}
		
		samples.push_back(sample);
	}
}

void PathEvaluator::planAndEvaluatePaths(const std::vector<double> &start_state, 
		                                 Eigen::MatrixXd &P_t,
					                     unsigned int &current_step,
		                                 double &timeout,		                                 
		                                 unsigned int &num_threads,
										 std::shared_ptr<shared::PathPlannerOptions> &path_planner_options) {	
	std::queue<std::shared_ptr<shared::PathEvaluationResult>> queue;	
	boost::thread_group eval_group;	
	for (size_t i = 0; i < num_threads; i++) {
		eval_group.add_thread(new boost::thread(&PathEvaluator::eval_thread, 
				                                this, 
				                                queue, 
				                                start_state,
				                                P_t,
				                                current_step,
				                                timeout,
												path_planner_options));		
	}
	
	sleep(timeout);
	eval_group.interrupt_all();
}

void PathEvaluator::eval_thread(std::queue<std::shared_ptr<shared::PathEvaluationResult>> &queue,		                        
		                        const std::vector<double> &start_state,
		                        Eigen::MatrixXd &P_t,
		                        unsigned int &current_step,
		                        double &planning_timeout,
								std::shared_ptr<shared::PathPlannerOptions> &path_planner_options) {
	std::unique_ptr<shared::DynamicPathPlanner> dynamic_path_planner;
			
	while (true) {
		try {
			//dynamic_path_planner = nullptr;
			dynamic_path_planner = shared::makeDynamicPathPlanner(robot_environment_, path_planner_options);	
			//construct a path
			cout << "plan..." << endl;	
			std::vector<std::vector<double>> solution = dynamic_path_planner->solve(start_state, planning_timeout);
			cout << "solution found: " << solution.size() << endl;
			
			//evaluate it
			/**double objective = evaluatePath(solution[0], 
					                        solution[1], 
					                        solution[2],
					                        solution[3],
					                        P_t,
					                        current_step);*/
			//put result in the queue
			mtx_.lock();
					
			mtx_.unlock();			
			
		}
		catch (boost::thread_interrupted&) {					
		}
	}
}

}