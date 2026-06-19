## create your own custom exception handling 
import sys
import logging
def get_error_details(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_meassage="Error occured in the python scrip name[{0}],linenumber[{1}],error message[{2}]".format(
    file_name,exc_tb.tb_lineno,str(error))
    return error_meassage
    ## create a class for customexception
class CustomException(Exception):
        def __init__(self,error_message,error_detail):
            super().__init__(error_message)
            ## create the objects
            self.error_message=get_error_details(error_message,error_detail)
            
        def __str__(self):
            return self.error_message


    
            