from predictionFile import Prediction
from trainQnAModel import ModelTraining

if __name__ == "__main__":

    # # For model Training
    # mdlTrnng = ModelTraining()
    # mdlTrnng.executeProcessing()

    # For model Prediction
    print('-------------------------------------------------------------------------------------------')
    print('Custom User Queries (Make sure there are spaces before each word)')
    print('-------------------------------------------------------------------------------------------')

    # Sample Data
    # user_story = "John travelled to the hallway . Mary journeyed to the bathroom ."
    # user_query = "Where is John ?"

    print('Please input a story')
    user_story = input().split(' ')
    print('Please input a query')
    user_query = input().split(' ')

    predctnObj = Prediction()
    prediction = predctnObj.executeProcessing(user_story, user_query)

    print('===============================Result================================')
    print(' '.join(user_story), ' '.join(user_query), '| Prediction:', prediction)
