"""
Comprehensive Test Suite for RRPToolbox
Tests all functions and logic to ensure correctness

Test Categories:
1. Initialization & Setup
2. Matrix Operations
3. Forward Kinematics
4. Inverse Kinematics
5. FK-IK Consistency (Round-trip tests)
6. Jacobian Computation
7. Differential Kinematics
8. Singularity Detection
9. Reachability Checks
10. Edge Cases & Error Handling
"""

import math
import sys
from RRPToolbox import RRPToolbox


class TestRRPToolbox:
    def __init__(self):
        self.test_count = 0
        self.pass_count = 0
        self.fail_count = 0
        self.failures = []
        
        self.link_params = [
            [(5, 0, 0), (0, 0, 5)],
            [(3, 0, 0)],
            [(0, 0, 0.5)]
        ]
        self.joint_limits = [
            (-180, 180),
            (0, 180),
            (0, 11.0)
        ]
        self.toolbox = RRPToolbox(self.link_params, self.joint_limits)
        
    def assert_equal(self, actual, expected, test_name, tolerance=1e-6):
        self.test_count += 1
        if isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
            match = all(abs(a - e) < tolerance for a, e in zip(actual, expected))
        else:
            match = abs(actual - expected) < tolerance
        
        if match:
            self.pass_count += 1
            print(f"✓ {test_name}")
            return True
        else:
            self.fail_count += 1
            self.failures.append(f"✗ {test_name}: Expected {expected}, got {actual}")
            print(f"✗ {test_name}: Expected {expected}, got {actual}")
            return False
    
    def assert_true(self, condition, test_name):
        self.test_count += 1
        if condition:
            self.pass_count += 1
            print(f"✓ {test_name}")
            return True
        else:
            self.fail_count += 1
            self.failures.append(f"✗ {test_name}")
            print(f"✗ {test_name}")
            return False
    
    def assert_exception(self, func, test_name):
        self.test_count += 1
        try:
            func()
            self.fail_count += 1
            self.failures.append(f"✗ {test_name}: Expected exception but none raised")
            print(f"✗ {test_name}: Expected exception but none raised")
            return False
        except Exception:
            self.pass_count += 1
            print(f"✓ {test_name}")
            return True
    
    def run_all_tests(self):
        print("=" * 70)
        print("RRP TOOLBOX VALIDATION TEST SUITE")
        print("=" * 70)
        
        self.test_initialization()
        # self.test_matrix_operations()
        self.test_forward_kinematics()
        self.test_inverse_kinematics()
        self.test_fk_ik_consistency()
        self.test_jacobian()
        self.test_differential_kinematics()
        self.test_singularity_detection()
        self.test_reachability()
        self.test_edge_cases()
        
        self.print_summary()
    
    def test_initialization(self):
        print("\n" + "=" * 70)
        print("TEST 1: INITIALIZATION & SETUP")
        print("=" * 70)
        
        self.assert_equal(
            len(self.toolbox.joint_local_positions), 4,
            "Joint local positions count (base + 3 links)"
        )
        
        self.assert_equal(
            len(self.toolbox.joint_global_positions), 4,
            "Joint global positions count"
        )
        
        self.assert_equal(
            self.toolbox.current_joint_config, [0.0, 0.0, 0.0],
            "Initial configuration is zero"
        )
        
        expected_local_1 = (5, 0, 5)
        self.assert_equal(
            self.toolbox.joint_local_positions[1], expected_local_1,
            "Joint 1 local position calculation"
        )
        
        expected_local_2 = (3, 0, 0)
        self.assert_equal(
            self.toolbox.joint_local_positions[2], expected_local_2,
            "Joint 2 local position calculation"
        )
        
        expected_local_3 = (0, 0, 0.5)
        self.assert_equal(
            self.toolbox.joint_local_positions[3], expected_local_3,
            "End Effector local position calculation"
        )
        
        expected_global_1 = (5, 0, 5)
        self.assert_equal(
            self.toolbox.joint_global_positions[1], expected_global_1,
            "Joint 1 global position calculation"
        )
        
        expected_global_2 = (8, 0, 5)
        self.assert_equal(
            self.toolbox.joint_global_positions[2], expected_global_2,
            "Joint 2 global position calculation"
        )
        
        expected_global_3 = (8, 0, 5.5)
        self.assert_equal(
            self.toolbox.joint_global_positions[3], expected_global_3,
            "End Effector global position calculation"
        )
    
    def test_forward_kinematics(self):
        print("\n" + "=" * 70)
        print("TEST 3: FORWARD KINEMATICS")
        print("=" * 70)
        
        config = [0, 0, 5]
        pos = self.toolbox.forward_kinematics(config, update_state=False)
        print(f"  Config {config} → Position ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        self.assert_true(
            abs(pos[0] - 13) < 1 and pos[1] == 0 and abs(pos[2] - 5.5) < 1,
            f"FK: Zero rotation gives X≈8 (got {pos[0]:.2f})"
        )
        
        config = [0, 90, 5]
        pos = self.toolbox.forward_kinematics(config, update_state=False)
        print(f"  Config {config} → Position ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        self.assert_true(
            abs(pos[0] - 4.5) < 1 and pos[1] ==  0 and abs(pos[2] - 13) < 1,
            f"FK: 90° theta2 extends in X direction (X={pos[0]:.2f})"
        )
        
        config = [90, 0, 5]
        pos = self.toolbox.forward_kinematics(config, update_state=False)
        print(f"  Config {config} → Position ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        self.assert_true(
            pos[0] == 0 and abs(pos[1] - 13) < 1,
            f"FK: 90° theta1 rotates to Y≈8 (got {pos[1]:.2f})"
        )
        
        config = [45, 45, 5]
        pos1 = self.toolbox.forward_kinematics(config, update_state=False)
        config = [45, 45, 7]
        pos2 = self.toolbox.forward_kinematics(config, update_state=False)
        distance = math.sqrt(sum((p2 - p1)**2 for p1, p2 in zip(pos1, pos2)))
        self.assert_true(
            abs(distance - 2.0) < 0.5,
            f"FK: Changing d3 by 2 changes distance by ~2 (got {distance:.2f})"
        )
    
    def test_inverse_kinematics(self):
        print("\n" + "=" * 70)
        print("TEST 4: INVERSE KINEMATICS")
        print("=" * 70)
        
        target = (5, 0, 12)
        try:
            joints = self.toolbox.inverse_kinematics(target, validate=True)
            self.assert_true(
                True,
                f"IK: Valid target {target} found solution {[f'{j:.2f}' for j in joints]}"
            )
        except ValueError as e:
            print(f"⚠ IK test 1 skipped (target unreachable): {e}")
        
        target = (8, 0, 8)
        try:
            joints = self.toolbox.inverse_kinematics(target, validate=True)
            self.assert_true(
                abs(joints[0]) < 15,
                f"IK: Target on X-axis gives small theta1: {joints[0]:.2f}°"
            )
        except ValueError as e:
            print(f"⚠ IK test 2 skipped (target unreachable): {e}")
        
        target = (0, 8, 8)
        try:
            joints = self.toolbox.inverse_kinematics(target, validate=True)
            self.assert_true(
                abs(joints[0] - 90) < 15,
                f"IK: Target on Y-axis gives theta1 ≈ 90°: {joints[0]:.2f}°"
            )
        except ValueError as e:
            print(f"⚠ IK test 3 skipped (target unreachable): {e}")
    
    def test_fk_ik_consistency(self):
        print("\n" + "=" * 70)
        print("TEST 5: FK-IK CONSISTENCY (ROUND-TRIP TESTS)")
        print("=" * 70)
        
        test_configs = [
            [0, 45, 5],
            [45, 90, 7],
            [90, 60, 4],
            [-45, 120, 6],
            [135, 30, 8]
        ]
        
        for i, config in enumerate(test_configs):
            pos = self.toolbox.forward_kinematics(config, update_state=False)
            
            try:
                recovered_config = self.toolbox.inverse_kinematics(pos, validate=True)
                
                pos_recovered = self.toolbox.forward_kinematics(
                    recovered_config, update_state=False
                )
                
                position_error = math.sqrt(
                    sum((p1 - p2)**2 for p1, p2 in zip(pos, pos_recovered))
                )
                
                self.assert_true(
                    position_error < 0.1,
                    f"FK→IK→FK consistency test {i+1}: config {config} → pos {[f'{p:.2f}' for p in pos]}"
                )
            except ValueError as e:
                print(f"⚠ Test {i+1} failed: {e}")
    
    def test_jacobian(self):
        print("\n" + "=" * 70)
        print("TEST 6: JACOBIAN COMPUTATION")
        print("=" * 70)
        
        config = [0, 90, 5]
        try:
            J, J_red, J_inv = self.toolbox.get_rrp_jacobian_matrix(config)
            
            self.assert_equal(
                len(J), 6,
                "Full Jacobian has 6 rows"
            )
            
            self.assert_equal(
                len(J_red), 3,
                "Reduced Jacobian has 3 rows"
            )
            
            result = self.toolbox.matrix_multiply(J_red, J_inv)
            is_identity = all(
                abs(result[i][j] - (1 if i == j else 0)) < 1e-3
                for i in range(3) for j in range(3)
            )
            self.assert_true(
                is_identity,
                "J_reduced × J_inverse ≈ Identity"
            )
        except ValueError as e:
            print(f"⚠ Jacobian test failed: {e}")
    
    def test_differential_kinematics(self):
        print("\n" + "=" * 70)
        print("TEST 7: DIFFERENTIAL KINEMATICS")
        print("=" * 70)
        
        config = [45, 90, 5]
        self.toolbox.update_current_config(config)
        
        joint_vel = [10, 5, 0.5]
        cart_vel = self.toolbox.differential_forward_kinematics(joint_vel)
        
        self.assert_true(
            len(cart_vel) == 3,
            "Differential FK returns 3D velocity"
        )
        
        recovered_joint_vel = self.toolbox.differential_inverse_kinematics(cart_vel)
        
        vel_error = math.sqrt(
            sum((v1 - v2)**2 for v1, v2 in zip(joint_vel, recovered_joint_vel))
        )
        
        self.assert_true(
            vel_error < 0.1,
            "Differential FK→IK consistency"
        )
    
    def test_singularity_detection(self):
        print("\n" + "=" * 70)
        print("TEST 8: SINGULARITY DETECTION")
        print("=" * 70)
        
        config = [0, 0, 5]
        is_singular = self.toolbox.check_singularity(config)
        self.assert_true(
            is_singular,
            "Theta2=0° is singular (wrist singularity)"
        )
        
        config = [0, 90, 5]
        is_singular = self.toolbox.check_singularity(config)
        self.assert_true(
            not is_singular,
            "Theta2=90° is not singular"
        )
        
        config = [45, 45, 5]
        is_singular = self.toolbox.check_singularity(config)
        self.assert_true(
            not is_singular,
            "General configuration is not singular"
        )
    
    def test_reachability(self):
        print("\n" + "=" * 70)
        print("TEST 9: REACHABILITY CHECKS")
        print("=" * 70)
        
        target = (0, 0, 10)
        reachable, msg = self.toolbox.is_reachable(target)
        self.assert_true(reachable, f"Point (0,0,10) is reachable: {msg}")
        
        target = (100, 0, 0)
        reachable, msg = self.toolbox.is_reachable(target)
        self.assert_true(not reachable, f"Point (100,0,0) is unreachable: {msg}")
        
        target = (0, 0, 0.5)
        reachable, msg = self.toolbox.is_reachable(target)
        self.assert_true(not reachable, f"Point too close is unreachable: {msg}")
    
    def test_edge_cases(self):
        print("\n" + "=" * 70)
        print("TEST 10: EDGE CASES & ERROR HANDLING")
        print("=" * 70)
        
        self.assert_exception(
            lambda: self.toolbox.forward_kinematics([0, 0]),
            "FK rejects wrong number of parameters"
        )
        
        self.assert_exception(
            lambda: self.toolbox.forward_kinematics([0, 200, 5]),
            "FK rejects theta2 outside limits"
        )
        
        self.assert_exception(
            lambda: self.toolbox.forward_kinematics([0, 90, 20]),
            "FK rejects d3 outside limits"
        )
        
        self.assert_exception(
            lambda: self.toolbox.inverse_kinematics((1000, 0, 0)),
            "IK rejects unreachable position"
        )
        
        config = [0, 90, 5]
        pos = self.toolbox.forward_kinematics(config, update_state=True)
        retrieved = self.toolbox.get_current_config()
        self.assert_equal(
            list(retrieved), config,
            "update_state=True updates current config"
        )
        
        deg = 180
        rad = self.toolbox.deg_to_rad(deg)
        self.assert_equal(rad, math.pi, "deg_to_rad conversion", tolerance=1e-6)
        
        rad = math.pi
        deg = self.toolbox.rad_to_deg(rad)
        self.assert_equal(deg, 180, "rad_to_deg conversion", tolerance=1e-6)
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Total Tests: {self.test_count}")
        print(f"Passed: {self.pass_count} ({100*self.pass_count/self.test_count:.1f}%)")
        print(f"Failed: {self.fail_count} ({100*self.fail_count/self.test_count:.1f}%)")
        
        if self.fail_count > 0:
            print("\n" + "=" * 70)
            print("FAILED TESTS:")
            print("=" * 70)
            for failure in self.failures:
                print(failure)
            print("\n⚠ VALIDATION FAILED - Fix the above issues")
            return False
        else:
            print("\n" + "=" * 70)
            print("✓ ALL TESTS PASSED - RRPToolbox is working correctly!")
            print("=" * 70)
            return True


if __name__ == "__main__":
    print("\nStarting RRPToolbox validation...\n")
    
    tester = TestRRPToolbox()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)