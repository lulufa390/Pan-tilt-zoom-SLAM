//
//  ParameterParser.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-10-12.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "dt_param_parser.h"
#include <assert.h>

namespace dt {
    ParameterParser::ParameterParser()
    {
        
    }
    ParameterParser::~ParameterParser()
    {
        
    }
    
    static void remove_end_return_symbol(FILE *pf)
    {
        char dummy;
        int ret_num = fscanf(pf, "%c", &dummy);
        assert(ret_num == 1);
        assert(dummy == '\n');
    }
    bool ParameterParser::loadParameter(const char *filename)
    {
        assert(filename);
        FILE *pf = fopen(filename, "r");
        if (!pf) {
            printf("can not open file: %s\n", filename);
            return false;
        }
        
        int_values_.clear();
        float_values_.clear();
        while (1) {
            char symbol = 't';
            int ret_num = fscanf(pf, "%c", &symbol);
            assert(ret_num == 1);
            // end of file
            if (symbol == 'E') {
                break;
            }
            else if(symbol == 'F')
            {
                // float sequence data
                char buf[1024] = {NULL};
                int num = 0;
                ret_num = fscanf(pf, "%s %d", buf, &num);
                assert(ret_num == 2);
                vector<double> values;
                for (int i = 0; i<num; i++) {
                    double val = 0;
                    ret_num = fscanf(pf, "%lf", &val);
                    assert(ret_num == 1);
                    values.push_back(val);
                }
                assert(values.size() == num);
                float_values_[string(buf)] = values;
                // remove \n
                remove_end_return_symbol(pf);
            }
            else if (symbol == 'I')
            {
                // integer sequence data
                char buf[1024] = {NULL};
                int num = 0;
                ret_num = fscanf(pf, "%s %d", buf, &num);
                assert(ret_num == 2);
                vector<int> values;
                for (int i = 0; i<num; i++) {
                    int val = 0;
                    ret_num = fscanf(pf, "%d", &val);
                    assert(ret_num == 1);
                    values.push_back(val);
                }
                assert(values.size() == num);
                int_values_[string(buf)] = values;
                // remove \n
                remove_end_return_symbol(pf);
            }
            else {
                printf("Error: symbol %c is not defined.\n", symbol);
                assert(0);
            }
        }
        fclose(pf);
        return true;
    }
    
    bool ParameterParser::getIntValue(const string & name, int & value) const
    {
        auto ite = int_values_.find(name);
        if (ite != int_values_.end()) {
            vector<int> values = ite->second;
            assert(values.size() == 1);
            value = values[0];
            return true;
        }
        else {
            printf("Error: can not find parameter %s from parser \n", name.c_str());
            assert(0);
        }
        return false;
    }
    
    bool ParameterParser::getBoolValue(const string & name, bool & value) const
    {
        auto ite = int_values_.find(name);
        if (ite != int_values_.end()) {
            vector<int> values = ite->second;
            assert(values.size() == 1);
            value = (values[0] != 0);
            return true;
        }
        else {
            printf("Error: can not find parameter %s\n", name.c_str());
            assert(0);
        }
    }
    
    bool ParameterParser::getFloatValue(const string & name, double & value) const
    {
        auto ite = float_values_.find(name);
        if (ite != float_values_.end()) {
            vector<double> values = ite->second;
            assert(values.size() == 1);
            value = values[0];
            return true;
        }
        else {
            printf("Error: can not find parameter %s\n", name.c_str());
            assert(0);
        }
        return false;
    }
    
    void ParameterParser::setIntValue(const string & name, int value)
    {
        vector<int> values;
        values.push_back(value);
        int_values_[name] = values;
    }
    
    void ParameterParser::setBoolValue(const string & name, bool value)
    {
        vector<int> values;
        values.push_back(value);
        int_values_[name] = values;
    }
    
    void ParameterParser::setFloatValue(const string & name, double value)
    {
        vector<double> values;
        values.push_back(value);
        float_values_[name] = values;
    }
    
    void ParameterParser::clean()
    {
        int_values_.clear();
        float_values_.clear();
    }
    
    void ParameterParser::printSelf() const
    {
        for (auto ite = int_values_.begin(); ite != int_values_.end(); ite ++) {
            string name = ite->first;
            vector<int> values = ite->second;
            printf("%s ", name.c_str());
            for (int i = 0; i<values.size(); i++) {
                printf("%d ", values[i]);
                if (i + 1== values.size()) {
                    printf("\n");
                }
            }
        }
        
        for (auto ite = float_values_.begin(); ite != float_values_.end(); ite ++) {
            string name = ite->first;
            vector<double> values = ite->second;
            printf("%s ", name.c_str());
            for (int i = 0; i<values.size(); i++) {
                printf("%lf ", values[i]);
                if (i + 1== values.size()) {
                    printf("\n");
                }
            }
        }
    }
    
    void ParameterParser::writeToFile(FILE *pf) const
    {
        assert(pf);
        for (auto ite = int_values_.begin(); ite != int_values_.end(); ite ++) {
            string name = ite->first;
            vector<int> values = ite->second;
            fprintf(pf, "I %s %lu ", name.c_str(), values.size());
            for (int i = 0; i<values.size(); i++) {
                if (i < values.size() - 1) {
                    fprintf(pf, "%d ", values[i]);
                }
                else {
                    fprintf(pf, "%d\n", values[i]);
                }
            }
        }
        
        for (auto ite = float_values_.begin(); ite != float_values_.end(); ite ++) {
            string name = ite->first;
            vector<double> values = ite->second;
            fprintf(pf, "F %s %lu", name.c_str(), values.size());
            for (int i = 0; i<values.size(); i++) {
                if (i < values.size() - 1) {
                    fprintf(pf, "%lf ", values[i]);
                }
                else {
                    fprintf(pf, "%lf\n", values[i]);
                }
            }
        }
        fprintf(pf, "E \n");
    }
    
    bool ParameterParser::readFromFile(FILE *pf)
    {
        int_values_.clear();
        float_values_.clear();
        while (1) {
            char symbol = 't';
            int ret_num = fscanf(pf, "%c", &symbol);
            assert(ret_num == 1);
            // end of file
            if (symbol == 'E') {
                break;
            }
            else if(symbol == 'F')
            {
                // float sequence data
                char buf[1024] = {NULL};
                int num = 0;
                ret_num = fscanf(pf, "%s %d", buf, &num);
                assert(ret_num == 2);
                vector<double> values;
                for (int i = 0; i<num; i++) {
                    double val = 0;
                    ret_num = fscanf(pf, "%lf", &val);
                    assert(ret_num == 1);
                    values.push_back(val);
                }
                assert(values.size() == num);
                float_values_[string(buf)] = values;
                // remove \n
                remove_end_return_symbol(pf);
            }
            else if (symbol == 'I')
            {
                // integer sequence data
                char buf[1024] = {NULL};
                int num = 0;
                ret_num = fscanf(pf, "%s %d", buf, &num);
                assert(ret_num == 2);
                vector<int> values;
                for (int i = 0; i<num; i++) {
                    int val = 0;
                    ret_num = fscanf(pf, "%d", &val);
                    assert(ret_num == 1);
                    values.push_back(val);
                }
                assert(values.size() == num);
                int_values_[string(buf)] = values;
                // remove \n
                remove_end_return_symbol(pf);
            }
            else {
                printf("Error: symbol %c is not defined.\n", symbol);
                assert(0);
            }
        }
        return true;
    }
    
} // namespace dt

