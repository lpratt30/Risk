import unittest
from unittest.mock import patch, Mock
import numpy as np

from atomic_actions import get_dice_bag, attack_territory


class TestGetDiceBag(unittest.TestCase):

    @patch('atomic_actions.np.random.randint')
    def test_dice_bag_size(self, mock_randint):
        mock_randint.return_value = np.array([1, 2, 3, 4, 5, 6])
        attacker_troops = 3
        defender_troops = 3
        attacker_rolls, defender_rolls, bag_size = get_dice_bag(attacker_troops, defender_troops)
        self.assertEqual(bag_size, attacker_troops + defender_troops)
        self.assertEqual(len(attacker_rolls), attacker_troops)
        self.assertEqual(len(defender_rolls), defender_troops)

    @patch('atomic_actions.np.random.randint')
    def test_attacker_rolls_sorted(self, mock_randint):
        mock_randint.return_value = np.array([6, 2, 3, 4, 5, 1])
        attacker_troops = 3
        defender_troops = 3
        attacker_rolls, defender_rolls, bag_size = get_dice_bag(attacker_troops, defender_troops)
        self.assertEqual(list(attacker_rolls), [6, 3, 2])
        self.assertEqual(list(defender_rolls), [5, 4, 1])

    @patch('atomic_actions.np.random.randint')
    def test_defender_rolls_sorted(self, mock_randint):
        mock_randint.return_value = np.array([1, 2, 3, 4, 5, 6])
        attacker_troops = 3
        defender_troops = 3
        attacker_rolls, defender_rolls, bag_size = get_dice_bag(attacker_troops, defender_troops)
        self.assertEqual(list(defender_rolls), [5, 4, 6])

    @patch('atomic_actions.np.random.randint')
    def test_return_structure(self, mock_randint):
        mock_randint.return_value = np.array([1, 2, 3, 4, 5, 6])
        attacker_troops = 3
        defender_troops = 3
        attacker_rolls, defender_rolls, bag_size = get_dice_bag(attacker_troops, defender_troops)
        self.assertIsInstance(attacker_rolls, np.ndarray)
        self.assertIsInstance(defender_rolls, np.ndarray)
        self.assertIsInstance(bag_size, int)


class TestAttackTerritory(unittest.TestCase):

    def setUp(self):
        self.from_territory = Mock()
        self.to_territory = Mock()

        self.from_territory.name = "A"
        self.to_territory.name = "B"

        self.from_territory.neighbor_names = ["B"]
        self.from_territory.owner.key = 1
        self.to_territory.owner.key = 2

        self.from_territory.troop_count = 4
        self.to_territory.troop_count = 2

        self.from_territory.owner.territories = {self.from_territory.name: 1}
        self.to_territory.owner.territories = {self.to_territory.name: 1}
        self.from_territory.owner.territory_count = 1
        self.to_territory.owner.territory_count = 1

        self.from_territory.owner.damage_dealt = {2: 0}
        self.to_territory.owner.damage_received = {1: 0}
        self.from_territory.owner.total_troops = 5
        self.to_territory.owner.total_troops = 3

    @patch('atomic_actions.get_dice_bag')
    def test_non_adjacent_territory(self, mock_get_dice_bag):
        self.from_territory.neighbor_names = ["C"]  # B is not a neighbor
        troops_lost_attacker, troops_lost_defender, attacker_won, is_legal = attack_territory(
            self.from_territory, self.to_territory, 4)
        self.assertFalse(is_legal)
        self.assertEqual(troops_lost_attacker, 0)
        self.assertEqual(troops_lost_defender, 0)
        self.assertFalse(attacker_won)

    @patch('atomic_actions.get_dice_bag')
    def test_attack_own_territory(self, mock_get_dice_bag):
        self.to_territory.owner.key = 1  # Same owner as from_territory
        troops_lost_attacker, troops_lost_defender, attacker_won, is_legal = attack_territory(
            self.from_territory, self.to_territory, 4)
        self.assertFalse(is_legal)
        self.assertEqual(troops_lost_attacker, 0)
        self.assertEqual(troops_lost_defender, 0)
        self.assertFalse(attacker_won)

    @patch('atomic_actions.get_dice_bag')
    def test_not_enough_troops_to_attack(self, mock_get_dice_bag):
        self.from_territory.troop_count = 1  # Not enough troops to attack
        troops_lost_attacker, troops_lost_defender, attacker_won, is_legal = attack_territory(
            self.from_territory, self.to_territory, 1)
        self.assertFalse(is_legal)
        self.assertEqual(troops_lost_attacker, 0)
        self.assertEqual(troops_lost_defender, 0)
        self.assertFalse(attacker_won)

    @patch('atomic_actions.get_dice_bag')
    def test_successful_attack(self, mock_get_dice_bag):
        mock_get_dice_bag.return_value = (np.array([6, 5, 4]), np.array([3, 2]), 5)
        troops_lost_attacker, troops_lost_defender, attacker_won, is_legal = attack_territory(
            self.from_territory, self.to_territory, 3)
        self.assertTrue(is_legal)
        self.assertEqual(troops_lost_attacker, 0)  # Attackers won
        self.assertEqual(troops_lost_defender, 2)  # All defender troops lost
        self.assertTrue(attacker_won)

    @patch('atomic_actions.get_dice_bag')
    def test_failed_attack_with_three(self, mock_get_dice_bag):
        mock_get_dice_bag.return_value = (np.array([1, 1, 1]), np.array([6, 5]), 5)
        troops_lost_attacker, troops_lost_defender, attacker_won, is_legal = attack_territory(
            self.from_territory, self.to_territory, 3)
        self.assertTrue(is_legal)
        self.assertEqual(troops_lost_attacker, 3)  # All attacker troops lost
        self.assertEqual(troops_lost_defender, 0)  # No defender troops lost
        self.assertFalse(attacker_won)

    @patch('atomic_actions.get_dice_bag')
    def test_failed_attack_with_two(self, mock_get_dice_bag):
        self.from_territory.troop_count = 3
        mock_get_dice_bag.return_value = (np.array([1, 1]), np.array([6, 5]), 5)
        troops_lost_attacker, troops_lost_defender, attacker_won, is_legal = attack_territory(
            self.from_territory, self.to_territory, 2)
        self.assertTrue(is_legal)
        self.assertEqual(troops_lost_attacker, 2)  # All attacker troops lost
        self.assertEqual(troops_lost_defender, 0)  # No defender troops lost
        self.assertFalse(attacker_won)

if __name__ == '__main__':
    unittest.main()
