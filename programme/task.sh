python3 main.py

red='\033[0;31m'
reset='\033[0m'

echo "\n \t \t \t \t ${red} ATTENTION TUTOR ${reset}\n"


echo "
Hello their (^ _^)/
    here are a couple notes about my submission
        - if you want to see all the bouding boxes in each step, hop insed
        Image.py and turn debug to True

        - if you want to see the kind of results I got from the trainning images
        run this command which is going to run the test unit of the provided
        trainning data from blackboard

            python3 -m unittest test_image

        -if the trainning data is different to the one which you have provided
        us on blackboard, you will need to delte the binary files with the names
        kNN_classfy, and kNN_lables so the algorithm can create new versions
        of these files to clasfy

        - also if you want to change directory on were to tet the image
        just jump into the main and change the test_path variable
"
