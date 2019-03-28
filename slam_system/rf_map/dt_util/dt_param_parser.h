//
//  ParameterParser.h
//  Classifer_RF
//
//  Created by jimmy on 2016-10-12.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__ParameterParser__
#define __Classifer_RF__ParameterParser__

#include <stdio.h>

#include <unordered_map>
#include <vector>
#include <string>

using std::unordered_map;
using std::vector;
using std::string;

namespace dt {
    //
    // example parameter
    /*
     I tree_depth 1 10
     F alpha 1 1.0
     I feature_steps 2 40 80
     I label_steps   1 40
     E
     */
    class ParameterParser
    {
    private:
        unordered_map<string, vector<int> > int_values_;
        unordered_map<string, vector<double> > float_values_;
        
    public:
        ParameterParser();
        ~ParameterParser();
        
        bool loadParameter(const char *filename);
        
        // single value
        bool getIntValue(const string & name, int & value) const;
        bool getBoolValue(const string & name, bool & value) const;
        bool getFloatValue(const string & name, double & value) const;
        
        //
        void setIntValue(const string & name, int value);
        void setBoolValue(const string & name, bool value);
        void setFloatValue(const string & name, double value);
        
        void clean();
        void printSelf() const;
        
        void writeToFile(FILE *pf) const;
        bool readFromFile(FILE *pf);
    };
} // namespace dt

#endif /* defined(__Classifer_RF__ParameterParser__) */
