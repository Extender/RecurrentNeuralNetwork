QT += core
QT -= gui

TARGET = RecurrentNeuralNetwork
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    rnn.cpp \
    io.cpp \
    text.cpp \
    rnnstate.cpp

HEADERS += \
    rnn.h \
    io.h \
    text.h \
    rnnstate.h

