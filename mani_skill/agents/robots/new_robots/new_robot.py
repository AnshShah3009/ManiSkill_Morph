import copy

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

# Helper function for deep copying controller configs
# def deepcopy_dict(d):
#     return copy.deepcopy(d)

@register_agent()
class FlexiArm(BaseAgent):
    """A flexible 7-DOF robot arm with high dexterity - Fixed for PickCube task"""
    uid = "flexi_arm"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/flexi_arm/flexi_arm.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2, dynamic_friction=2, restitution=0.0)
        ),
        link=dict(
            Fixed_Jaw=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            Moving_Jaw=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0, 0.2, 0, -0.8, 0, 0.6, 0, 0.08]),  # Increased gripper opening
            pose=sapien.Pose(q=euler2quat(0, 0, 0)),
        ),
        candle=Keyframe(
            qpos=np.array([0, -1.0, 0, -0.5, 0, -0.5, 0, 0.08]),  # Increased gripper opening
            pose=sapien.Pose(q=euler2quat(0, 0, 0)),
        ),
        zero=Keyframe(
            qpos=np.array([0.0] * 7 + [0.08]),  # Better default gripper position
            pose=sapien.Pose(q=euler2quat(0, 0, 0)),
        ),
        # Add pick-specific keyframes
        pick_ready=Keyframe(
            qpos=np.array([0, 0.5, 0.2, -1.2, 0, 0.8, 0, 0.08]),  # Good picking pose
            pose=sapien.Pose(q=euler2quat(0, 0, 0)),
        ),
    )

    @property
    def _controller_configs(self):
        pd_joint_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            lower=None,
            upper=None,
            stiffness=[1e3] * 8,
            damping=[1e2] * 8,
            force_limit=60,
            normalize_action=False,
        )

        # Fixed delta control ranges - larger gripper range for proper grasping
        pd_joint_delta_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            [-0.05] * 7 + [-0.12],  # Arm joints + gripper closing
            [0.05] * 7 + [0.36],    # Arm joints + gripper opening
            stiffness=[1e3] * 7 + [5e2],  # Slightly softer gripper control
            damping=[1e2] * 7 + [50],     # Lower damping for gripper
            force_limit=[60] * 7 + [30],  # Lower force limit for gripper
            use_delta=True,
            use_target=False,
        )

        pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_delta_pos=pd_joint_delta_pos,
            pd_joint_pos=pd_joint_pos,
            pd_joint_target_delta_pos=pd_joint_target_delta_pos,
        )
        return deepcopy_dict(controller_configs)

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.finger1_link = self.robot.links_map["Fixed_Jaw"]
        self.finger2_link = self.robot.links_map["Moving_Jaw"]
        self.finger1_tip = self.robot.links_map["Fixed_Jaw_tip"]
        self.finger2_tip = self.robot.links_map["Moving_Jaw_tip"]

    @property
    def tcp_pos(self):
        return (self.finger1_tip.pose.p + self.finger2_tip.pose.p) / 2

    @property
    def tcp_pose(self):
        return Pose.create_from_pq(self.tcp_pos, self.finger1_link.pose.q)

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Fixed grasping detection for cube picking"""
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # Fixed: Use proper gripper directions based on jaw movement
        # For Z-axis revolute gripper, the contact should be perpendicular to Z
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]  # Y direction
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]  # -Y direction
        
        # Handle zero force cases
        l_contact_forces_safe = l_contact_forces.clone()
        r_contact_forces_safe = r_contact_forces.clone()
        l_contact_forces_safe[lforce < 1e-6] = ldirection[lforce < 1e-6]
        r_contact_forces_safe[rforce < 1e-6] = rdirection[rforce < 1e-6]
        
        langle = common.compute_angle_between(ldirection, l_contact_forces_safe)
        rangle = common.compute_angle_between(rdirection, r_contact_forces_safe)
        
        # More lenient force threshold and angle for cube grasping
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        
        # Alternative: also consider if gripper is closed enough
        gripper_pos = self.robot.get_qpos()[:, -1]  # Last joint is gripper
        gripper_closed = gripper_pos < 0.04  # Gripper is reasonably closed
        
        # Grasping if both force conditions OR gripper is closed with some force
        force_grasp = torch.logical_and(lflag, rflag)
        weak_grasp = torch.logical_and(gripper_closed, torch.logical_or(lforce > 0.1, rforce > 0.1))
        
        return torch.logical_or(force_grasp, weak_grasp)

    def is_static(self, threshold=0.2):
        qvel = self.robot.get_qvel()[:, :-1]  # exclude gripper joint
        return torch.max(torch.abs(qvel), 1)[0] <= threshold




# @register_agent()
# class FlexiArm(BaseAgent):
#     """A flexible 7-DOF robot arm with high dexterity"""
#     uid = "flexi_arm"
#     urdf_path = f"{PACKAGE_ASSET_DIR}/robots/flexi_arm/flexi_arm.urdf"
#     urdf_config = dict(
#         _materials=dict(
#             gripper=dict(static_friction=2, dynamic_friction=2, restitution=0.0)
#         ),
#         link=dict(
#             Fixed_Jaw=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
#             Moving_Jaw=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
#         ),
#     )

#     keyframes = dict(
#         rest=Keyframe(
#             qpos=np.array([0, 0.2, 0, -0.8, 0, 0.6, 0, 0.02]),
#             pose=sapien.Pose(q=euler2quat(0, 0, 0)),
#         ),
#         candle=Keyframe(
#             qpos=np.array([0, -1.0, 0, -0.5, 0, -0.5, 0, 0.02]),
#             pose=sapien.Pose(q=euler2quat(0, 0, 0)),
#         ),
#         zero=Keyframe(
#             qpos=np.array([0.0] * 8),
#             pose=sapien.Pose(q=euler2quat(0, 0, 0)),
#         ),
#     )

#     @property
#     def _controller_configs(self):
#         pd_joint_pos = PDJointPosControllerConfig(
#             [joint.name for joint in self.robot.active_joints],
#             lower=None,
#             upper=None,
#             stiffness=[1e3] * 8,
#             damping=[1e2] * 8,
#             force_limit=60,
#             normalize_action=False,
#         )

#         pd_joint_delta_pos = PDJointPosControllerConfig(
#             [joint.name for joint in self.robot.active_joints],
#             [-0.05] * 7 + [-0.05],  # Smaller deltas for more joints
#             [0.05] * 7 + [0.05],
#             stiffness=[1e3] * 8,
#             damping=[1e2] * 8,
#             force_limit=60,
#             use_delta=True,
#             use_target=False,
#         )

#         pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
#         pd_joint_target_delta_pos.use_target = True

#         controller_configs = dict(
#             pd_joint_delta_pos=pd_joint_delta_pos,
#             pd_joint_pos=pd_joint_pos,
#             pd_joint_target_delta_pos=pd_joint_target_delta_pos,
#         )
#         return deepcopy_dict(controller_configs)

#     def _after_loading_articulation(self):
#         super()._after_loading_articulation()
#         self.finger1_link = self.robot.links_map["Fixed_Jaw"]
#         self.finger2_link = self.robot.links_map["Moving_Jaw"]
#         self.finger1_tip = self.robot.links_map["Fixed_Jaw_tip"]
#         self.finger2_tip = self.robot.links_map["Moving_Jaw_tip"]

#     @property
#     def tcp_pos(self):
#         return (self.finger1_tip.pose.p + self.finger2_tip.pose.p) / 2

#     @property
#     def tcp_pose(self):
#         return Pose.create_from_pq(self.tcp_pos, self.finger1_link.pose.q)

#     def is_grasping(self, object: Actor, min_force=0.4, max_angle=115):
#         l_contact_forces = self.scene.get_pairwise_contact_forces(
#             self.finger1_link, object
#         )
#         r_contact_forces = self.scene.get_pairwise_contact_forces(
#             self.finger2_link, object
#         )
#         lforce = torch.linalg.norm(l_contact_forces, axis=1)
#         rforce = torch.linalg.norm(r_contact_forces, axis=1)

#         ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 2]
#         rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 2]
#         langle = common.compute_angle_between(ldirection, l_contact_forces)
#         rangle = common.compute_angle_between(rdirection, r_contact_forces)
        
#         lflag = torch.logical_and(
#             lforce >= min_force, torch.rad2deg(langle) <= max_angle
#         )
#         rflag = torch.logical_and(
#             rforce >= min_force, torch.rad2deg(rangle) <= max_angle
#         )
#         return torch.logical_and(lflag, rflag)

#     def is_static(self, threshold=0.2):
#         qvel = self.robot.get_qvel()[:, :-1]  # exclude gripper joint
#         return torch.max(torch.abs(qvel), 1)[0] <= threshold
    
@register_agent()
class CompactArm(BaseAgent):
    """A more compact 4-DOF robot arm"""
    uid = "compact_arm"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/compact_arm/compact_arm.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2, dynamic_friction=2, restitution=0.0)
        ),
        link=dict(
            Fixed_Jaw=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            Moving_Jaw=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0, 0.3, -0.6, 0, 0.03]),
            pose=sapien.Pose(q=euler2quat(0, 0, 0)),
        ),
        folded=Keyframe(
            qpos=np.array([0, 1.2, -2.0, 0, 0.03]),
            pose=sapien.Pose(q=euler2quat(0, 0, 0)),
        ),
        zero=Keyframe(
            qpos=np.array([0.0] * 5),
            pose=sapien.Pose(q=euler2quat(0, 0, 0)),
        ),
    )

    @property
    def _controller_configs(self):
        pd_joint_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            lower=None,
            upper=None,
            stiffness=[1e3] * 5,
            damping=[1e2] * 5,
            force_limit=80,
            normalize_action=False,
        )

        pd_joint_delta_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            [-0.1, -0.1, -0.1, -0.1, -0.02],
            [0.1, 0.1, 0.1, 0.1, 0.02],
            stiffness=[1e3] * 5,
            damping=[1e2] * 5,
            force_limit=80,
            use_delta=True,
            use_target=False,
        )

        pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_delta_pos=pd_joint_delta_pos,
            pd_joint_pos=pd_joint_pos,
            pd_joint_target_delta_pos=pd_joint_target_delta_pos,
        )
        return deepcopy_dict(controller_configs)

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.finger1_link = self.robot.links_map["Fixed_Jaw"]
        self.finger2_link = self.robot.links_map["Moving_Jaw"]
        self.finger1_tip = self.robot.links_map["Fixed_Jaw_tip"]
        self.finger2_tip = self.robot.links_map["Moving_Jaw_tip"]

    @property
    def tcp_pos(self):
        return (self.finger1_tip.pose.p + self.finger2_tip.pose.p) / 2

    @property
    def tcp_pose(self):
        return Pose.create_from_pq(self.tcp_pos, self.finger1_link.pose.q)

    def is_grasping(self, object: Actor, min_force=0.3, max_angle=120):
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 2]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 2]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold=0.15):
        qvel = self.robot.get_qvel()[:, :-1]  # exclude gripper joint
        return torch.max(torch.abs(qvel), 1)[0] <= threshold




# @register_agent()
# class ScorpionArm(BaseAgent):
#     """Fixed version of your scorpion arm robot"""
#     uid = "scorpion_arm"
#     urdf_path = f"{PACKAGE_ASSET_DIR}/robots/scorpion_arm/scorpion_arm.urdf"
#     urdf_config = dict(
#         _materials=dict(
#             gripper=dict(static_friction=2, dynamic_friction=2, restitution=0.0)
#         ),
#         link=dict(
#             Fixed_Jaw=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
#             Moving_Jaw=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
#         ),
#     )

#     keyframes = dict(
#         rest=Keyframe(
#             qpos=np.array([0, 0.5, -0.5, 0, 0, 0.02]),  # Better rest position
#             pose=sapien.Pose(q=euler2quat(0, 0, 0)),
#         ),
#         elevated=Keyframe(
#             qpos=np.array([0, -0.3, 0.8, 0.2, 0, 0.02]),
#             pose=sapien.Pose(q=euler2quat(0, 0, 0)),
#         ),
#         zero=Keyframe(
#             qpos=np.array([0.0] * 6),
#             pose=sapien.Pose(q=euler2quat(0, 0, 0)),
#         ),
#     )

#     @property
#     def _controller_configs(self):
#         pd_joint_pos = PDJointPosControllerConfig(
#             [joint.name for joint in self.robot.active_joints],
#             lower=None,
#             upper=None,
#             stiffness=[1e3] * 6,
#             damping=[1e2] * 6,
#             force_limit=100,
#             normalize_action=False,
#         )

#         pd_joint_delta_pos = PDJointPosControllerConfig(
#             [joint.name for joint in self.robot.active_joints],
#             [-0.1, -0.1, -0.1, -0.1, -0.1, -0.01],  # Delta limits
#             [0.1, 0.1, 0.1, 0.1, 0.1, 0.01],
#             stiffness=[1e3] * 6,
#             damping=[1e2] * 6,
#             force_limit=100,
#             use_delta=True,
#             use_target=False,
#         )

#         pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
#         pd_joint_target_delta_pos.use_target = True

#         controller_configs = dict(
#             pd_joint_delta_pos=pd_joint_delta_pos,
#             pd_joint_pos=pd_joint_pos,
#             pd_joint_target_delta_pos=pd_joint_target_delta_pos,
#         )
#         return deepcopy_dict(controller_configs)

#     def _after_loading_articulation(self):
#         super()._after_loading_articulation()
#         self.finger1_link = self.robot.links_map["Fixed_Jaw"]
#         self.finger2_link = self.robot.links_map["Moving_Jaw"]
#         self.finger1_tip = self.robot.links_map["Fixed_Jaw_tip"]
#         self.finger2_tip = self.robot.links_map["Moving_Jaw_tip"]

#     @property
#     def tcp_pos(self):
#         return (self.finger1_tip.pose.p + self.finger2_tip.pose.p) / 2

#     @property
#     def tcp_pose(self):
#         return Pose.create_from_pq(self.tcp_pos, self.finger1_link.pose.q)

#     def is_grasping(self, object: Actor, min_force=0.5, max_angle=110):
#         l_contact_forces = self.scene.get_pairwise_contact_forces(
#             self.finger1_link, object
#         )
#         r_contact_forces = self.scene.get_pairwise_contact_forces(
#             self.finger2_link, object
#         )
#         lforce = torch.linalg.norm(l_contact_forces, axis=1)
#         rforce = torch.linalg.norm(r_contact_forces, axis=1)

#         ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 2]
#         rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 2]
#         langle = common.compute_angle_between(ldirection, l_contact_forces)
#         rangle = common.compute_angle_between(rdirection, r_contact_forces)
        
#         lflag = torch.logical_and(
#             lforce >= min_force, torch.rad2deg(langle) <= max_angle
#         )
#         rflag = torch.logical_and(
#             rforce >= min_force, torch.rad2deg(rangle) <= max_angle
#         )
#         return torch.logical_and(lflag, rflag)

#     def is_static(self, threshold=0.2):
#         qvel = self.robot.get_qvel()[:, :-1]  # exclude gripper joint
#         return torch.max(torch.abs(qvel), 1)[0] <= threshold

