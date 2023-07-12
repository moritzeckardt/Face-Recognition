'''
Created on 06.01.2021
@author: Max, Charly
'''

import unittest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from Eigenfaces import process_and_train, calculate_average_face, calculate_eigenfaces, get_feature_representation, \
    reconstruct_image, classify_image


class TestEigenfaces(unittest.TestCase):
    def setUp(self) -> None:
        # load the dataset and extract dimensions
        lfw_dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        _, self.h, self.w = lfw_dataset.images.shape

        # extract samples and labels (images and names)
        X = lfw_dataset.data
        y = lfw_dataset.target
        self.labels = lfw_dataset.target_names

        # split in training & test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25)

    def test_process_and_train(self):
        u, num_eigenfaces, _ = process_and_train(self.y_train, self.X_train, self.X_train.shape[0], self.h, self.w)

        # test if SVD is performed
        self.assertTrue(np.allclose(np.dot(u, u.T), np.eye(len(u), len(u)), atol=0.5), msg='Did you perform a SVD?')
        # check number of eigenfaces
        self.assertEqual(self.X_train.shape[0]-1,  num_eigenfaces, msg='Check your number of eigenfaces, '
                                                                      'hint: jupyter-notebook')
        self.assertEqual(u.shape[0], num_eigenfaces, msg="Check the number of eigenfaces you return...")
        # check size of eigenfaces is the same as images
        self.assertEqual(u.shape[1], self.X_train.shape[1], msg='Check your number of eigenfaces, '
                                                                      'hint: jupyter-notebook')
    def test_average(self):
        # test values
        a = np.eye(10, 20)
        b = 5 * np.eye(20, 10)
        # results
        true_a = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        false_a = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        true_b = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

        # student's result
        avg_a = calculate_average_face(a)
        avg_b = calculate_average_face(b)

        self.assertIsInstance(a, np.ndarray, msg='Must return a 1D vector')
        self.assertNotEqual(false_a.shape, avg_a.shape, msg='Check your mean axis')
        self.assertTrue(np.allclose(avg_a, true_a, atol=0.1), msg='Check your average image')
        self.assertEqual(true_b.shape, avg_b.shape, msg='The images should be row vectors')
        self.assertTrue(np.allclose(avg_b, true_b, atol=0.1), msg='Check your average image')


    def test_calculate_eigenfaces(self):
        a = np.arange(72).reshape(8, 9)
        avg = np.ones(9)
        eigenface = calculate_eigenfaces(a, avg, 7, 3, 3)

        self.assertNotEqual(8, eigenface.shape[1], msg='Make sure to return same-sized images!')
        self.assertEqual(7, eigenface.shape[0], msg='Make sure to only return the first n wanted eigenfaces.')
        self.assertEqual((7, 9), eigenface.shape, msg='Check the shape of your eigenface matrix')
        self.assertTrue(np.allclose(np.dot(eigenface, eigenface.T), np.eye(len(eigenface)), atol=0.1),
                        msg='There is something wrong with your eigenface matrix -> check your SVD')

        ones = 5 * np.eye(9)
        avg_1 = np.zeros(9)
        eigenface_1 = calculate_eigenfaces(ones, avg_1, 9, 3, 3)
        self.assertTrue(np.allclose(np.eye(9), eigenface_1, atol=0.1), msg='Check your eigenface computation...')

        ones_2 = 5 * np.eye(9)
        avg_2 = np.ones(9)
        eigenface_2 = calculate_eigenfaces(ones_2, avg_2, 9, 3, 3)
        self.assertFalse(np.allclose(np.eye(9), eigenface_2, atol=0.1), msg='Do not forget to subtract the mean')

        u1 = np.array([[np.cos(30), -np.sin(30), 0], [np.sin(30), np.cos(30), 0], [0, 0, 1]])
        s1 = np.eye(3)*np.array([3, 2, 1])
        v1 = np.array([[np.cos(10), 0, -np.sin(10)], [0, 1, 0], [np.sin(10), 0, np.cos(10)]])

        eigenface_3 = calculate_eigenfaces((u1@s1@v1).T, np.zeros(3), 3, 1.5, 1.5)
        self.assertTrue(np.allclose(np.abs(u1.T), np.abs(eigenface_3), atol=0.1),
                        msg='Something wrong with your SVD or your choice of returned eigenfaces')

    def test_get_feature_representation(self):
        coeff_1 = get_feature_representation(5 * np.ones((6, 4)), np.ones((8, 4)), np.zeros(4), 7)
        self.assertEqual(6, coeff_1.shape[0], msg='Check the first dim of your coefficients')
        self.assertEqual(7, coeff_1.shape[1], msg='Check the second dim of your coefficients')
        self.assertTrue(np.allclose(20 * np.ones((6, 7)), coeff_1, atol=0.1), msg='Check your coefficient computation')

        coeff_2 = get_feature_representation(5 * np.ones((6, 4)), np.ones((8, 4)), np.ones(4), 7)
        self.assertTrue(np.allclose(16 * np.ones((6, 7)), coeff_2, atol=0.1), msg='Check your zero mean image')

    def test_reconstruct_image(self):
        reco_1 = reconstruct_image(np.ones(12), np.ones((6, 12)), np.zeros(12), 6, 3, 4)
        self.assertEqual(3, reco_1.shape[0], msg='Check the first dim of your reco_img')
        self.assertEqual(4, reco_1.shape[1], msg='Check the second dim of your reco_img')

        reco_2 = reconstruct_image(np.ones(12), np.zeros((6, 12)), np.ones(12), 6, 3, 4)
        self.assertTrue(np.allclose(np.ones((3, 4)), reco_2, atol=0.1),
                        msg='Check what array you are using as the starting point for your reconstruction')

        faces_3 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        reco_3_0 = reconstruct_image(np.ones(12), faces_3, np.zeros(12), 2, 3, 4)
        res_3_0 = 12 * np.ones(12)
        res_3_0[10] = 14
        res_3_0[11] = 14
        self.assertFalse(np.allclose(res_3_0.reshape(3, 4), reco_3_0, atol=0.1),
                         msg='Check that you are using the right number of eigenface; Seems like you are using too many')

        res_3_1 = np.zeros(12)
        res_3_1[10] = 2
        res_3_1[11] = 2
        reco_3_1 = reconstruct_image(np.ones(12), faces_3, np.zeros(12), 2, 3, 4)
        self.assertTrue(np.allclose(res_3_1.reshape(3, 4), reco_3_1, atol=0.1),
                        msg='Check that you are using the right number of eigenfaces')

        img_1 = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0])
        faces_4 = np.array([[0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
                           [0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
        reco_4 = reconstruct_image(img_1, faces_4, np.zeros(20), 4, 4, 5)
        res_4 = np.array([[8., 8., 12.,  8.,  8.],
                          [13., 13., 17., 13., 13.],
                          [8.,  8., 12.,  8.,  8.],
                          [8., 8., 12., 8., 8.]])
        self.assertTrue(np.allclose(res_4, reco_4, atol=0.3),
                        msg='Your weighted reconstruction does not work')

    def test_classification(self):
        # train the classifier.
        eigenfaces, num_eigenfaces, avg = process_and_train(self.y_train, self.X_train, self.X_train.shape[0], self.h, self.w)
        prediction = classify_image(np.copy(self.X_train[0]), np.copy(eigenfaces), np.copy(avg), num_eigenfaces, self.h, self.w)
        self.assertIsInstance(prediction, np.ndarray, msg='Must return a nd-array')

        # get the student predictions.
        predictions = np.zeros(self.X_test.shape[0])
        for i in range(len(predictions)):
            predictions[i] = classify_image(np.copy(self.X_test[i]), np.copy(eigenfaces), np.copy(avg), num_eigenfaces, self.h, self.w)

        # check the classifier performance. Passed if above 60% classification rate. (50% would be random guessing)
        report = classification_report(self.y_test, predictions, target_names=self.labels, output_dict=True)
        self.assertTrue(report['weighted avg']['precision'] > 0.6,
                        msg='Check your classification -> Your overall precision is too low')
        self.assertTrue(report['weighted avg']['recall'] > 0.6,
                        msg='Check your classification -> Your overall recall is too low')
        self.assertTrue(report['weighted avg']['f1-score'] > 0.6,
                        msg='Check your classification -> Your overall f1 score is too low')


if __name__ == '__main__':
    unittest.main()
