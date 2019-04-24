import argparse
import os
import torchvision
import torchvision.transforms as transforms
import torch
import classify_svhn
from classify_svhn import Classifier

import numpy as np
from itertools import tee
from scipy import linalg

SVHN_PATH = "svhn"
PROCESS_BATCH_SIZE = 32


def get_sample_loader(path, batch_size):
    """
    Loads data from `[path]/samples`

    - Ensure that path contains only one directory
      (This is due ot how the ImageFolder dataset loader
       works)
    - Ensure that ALL of your images are 32 x 32.
      The transform in this function will rescale it to
      32 x 32 if this is not the case.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    data = torchvision.datasets.ImageFolder(
        path,
        transform=transforms.Compose([
            transforms.Resize((32, 32), interpolation=2),
            classify_svhn.image_transform
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=2,
    )
    return data_loader


def get_test_loader(batch_size):
    """
    Downloads (if it doesn't already exist) SVHN test into
    [pwd]/svhn.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    testset = torchvision.datasets.SVHN(
        SVHN_PATH, split='test',
        download=True,
        transform=classify_svhn.image_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
    )
    return testloader


def extract_features(classifier, data_loader):
    """
    Iterator of features for each image.
    """
    with torch.no_grad():
        for x, _ in data_loader:
            h = classifier.extract_features(x).numpy()
            for i in range(h.shape[0]):
                yield h[i]


def compute_mean(iterator):

    sum = np.zeros(512,)
    count = 0
    for vector in iterator:
        sum += vector
        count += 1
    print('Mean       : Processed %d samples' % count)
    return sum/count


def compute_covariance(iterator,mean):

    sum = np.zeros((512,512))
    count = 0
    mean = np.reshape(mean,(512,1))
    for vector in iterator:
        difference = (np.reshape(vector,(512,1)) - mean)
        sum += np.matmul(difference,difference.T)
        count += 1
    print('Covariance : Processed %d samples' % count)
    return sum/count


def calculate_mean_covariance(iterator):
    iterator, iterator_copy = tee(iterator)
    mean = compute_mean(iterator)
    covariance = compute_covariance(iterator_copy, mean)
    return mean, covariance


def calculate_fid_score(sample_feature_iterator,
                        testset_feature_iterator):

    print("\nSample Iterator")
    print("----------------")
    sample_mean, sample_covariance = calculate_mean_covariance(sample_feature_iterator)

    print("\n\nTest Iterator")
    print("-------------")
    test_mean, test_covariance = calculate_mean_covariance(testset_feature_iterator)

    mean_distance = np.linalg.norm(sample_mean - test_mean)**2

    tolerance = 1e-6

    offset = np.eye(512,512) * tolerance

    sqrt_product = linalg.sqrtm(np.matmul(sample_covariance + offset, test_covariance + offset))

    # Handling ill-conditioning by allowing a negligible imaginary component of square root
    # Diagonal of imaginary component should be very close to 0

    if np.iscomplexobj(sqrt_product):
        if not np.allclose(np.diagonal(sqrt_product).imag, 0, atol=tolerance):
            print("Imaginary component error margin high")
            margin = np.max(np.abs(sqrt_product.imag))
            print("Margin of error = ",margin)
        sqrt_product = sqrt_product.real

    covariance_distance = (sample_covariance + test_covariance - 2 * sqrt_product).trace()

    score = mean_distance + covariance_distance

    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Score a directory of images with the FID score.')
    parser.add_argument('--model', type=str, default="svhn_classifier.pt",
                        help='Path to feature extraction model.')
    parser.add_argument('directory', type=str, default="./samples",
                        help='Path to image directory')
    args = parser.parse_args()

    quit = False
    if not os.path.isfile(args.model):
        print("Model file " + args.model + " does not exist.")
        quit = True
    if not os.path.isdir(args.directory):
        print("Directory " + args.directory + " does not exist.")
        quit = True
    if quit:
        exit()

    classifier = torch.load(args.model, map_location='cpu')
    classifier.eval()

    sample_loader = get_sample_loader(args.directory,
                                      PROCESS_BATCH_SIZE)
    sample_f = extract_features(classifier, sample_loader)

    test_loader = get_test_loader(PROCESS_BATCH_SIZE)
    test_f = extract_features(classifier, test_loader)

    fid_score = calculate_fid_score(sample_f, test_f)
    print("\n")
    print("FID score:", fid_score)
    print("\n")
