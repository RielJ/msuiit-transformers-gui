import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15

ApplicationWindow{
    id: window
    width: 1440
    height: 1024
    visible: true
    title: qsTr("Distribution Line Detection")
    color : "#12232E"

    // SET FLAGS
    //flags: Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.CustomizedWindowHint | Qt.MSWindowsFixedSizeDialogHint | Qt.WindowTitleHint

    Rectangle{
        id: topBar
        width: 1440
        height: 70
        color : "#177CC1"
        anchors{
            left: parent.left
            right: parent.right
            top: parent.top
        }

        Text{
            text: qsTr("DISTRIBUTION LINE DETECTION")
            anchors.verticalCenter: parent.verticalCenter
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text. AlignVCenter
            anchors.horizontalCenter: parent.horizontalCenter
            color: "#F5F7FA"
            font.pointSize: 20
            font.family: "Montserrat"
            font.bold : true
        }
       

    }

    Rectangle{
        id: body
        width: 1140
        height: 499
        color : "#12232E"
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.verticalCenter: paren.verticalCenter
        anchors.top: topBar.bottom 
        anchors.topMargin: 88
        
        Row{
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.verticalCenter: paren.verticalCenter

            spacing: 41

            Rectangle {
                id:videoOutput
                width: 729
                height: 499
                radius: 4
                color: "#1F3647"                 
            }

            Rectangle {
                id: appControls
                width: 370
                height: 499
                radius: 4
                color: "#12232E" 

                Column{
                    
                    spacing: 10

                    Label{
                        id: modeLabel
                        text: qsTr("Mode")
                        color: "#F5F7FA"
                        font.pointSize: 16
                        font.family: "Montserrat"
                        font.bold : true
                    }

                    Row{      
                        spacing: 0                
                        
                        Button{
                            width: 185
                            height: 41
                            background : Rectangle{
                                radius: 4
                                color: "#177CC1"
                            }
                            Text {
                                text: qsTr("Video")
                                anchors.verticalCenter: parent.verticalCenter
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text. AlignVCenter
                                anchors.horizontalCenter: parent.horizontalCenter
                                color: "#F5F7FA"
                                font.pointSize: 16
                                font.family: "Montserrat"
                                font.bold : true
                            }
                        }
                        Button{
                            width: 185
                            height: 41
                            background : Rectangle{
                                radius: 4
                                color: "#1F3647"
                            }
                            Text {
                                text: qsTr("Image")
                                anchors.verticalCenter: parent.verticalCenter
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text. AlignVCenter
                                anchors.horizontalCenter: parent.horizontalCenter
                                color: "#F5F7FA"
                                font.pointSize: 16
                                font.family: "Montserrat"
                                font.bold : true
                            }
                            
                        }

                    }

                    Label{
                        id: importVideoLabel
                        text: qsTr("Import Video")
                        color: "#F5F7FA"
                        font.pointSize: 16
                        font.family: "Montserrat"
                        font.bold : true
                    }

                    Button{
                            width: 370
                            height: 41
                            background : Rectangle{
                                radius: 4
                                color: "#177CC1"
                            }
                            Text {
                                text: qsTr("Upload")
                                anchors.verticalCenter: parent.verticalCenter
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text. AlignVCenter
                                anchors.horizontalCenter: parent.horizontalCenter
                                color: "#F5F7FA"
                                font.pointSize: 16
                                font.family: "Montserrat"
                                font.bold : true
                            }
                    }

                    Label{
                        id: detectionLabel
                        text: qsTr("Detection")
                        color: "#F5F7FA"
                        font.pointSize: 16
                        font.family: "Montserrat"
                        font.bold : true
                    }

                    ComboBox {                       
                        
                        id: comboBoxDetection                   
                        Text {
                            text: qsTr("Select Detection")
                            anchors.verticalCenter: parent.verticalCenter
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text. AlignVCenter
                            anchors.left: parent.left
                            leftPadding: 10
                            color: "#F5F7FA"
                            font.pointSize: 16
                            font.family: "Montserrat"
                            font.bold : false
                            }
                        background: Rectangle {
                            color:"#1F3647"
                        }
                        width: 370
                        height: 41
                        model: ListModel {
                            ListElement { text: "Select Detection" }
                            ListElement { text: "Powerline" }
                            ListElement { text: "Component" }
                            ListElement { text: "Obstruction" }
                        }
                    }

                    Label{
                        id: componentsLabel
                        text: qsTr("Components")
                        color: "#F5F7FA"
                        font.pointSize: 16
                        font.family: "Montserrat"
                        font.bold : true
                    }
                        
                    Column{
                        CheckBox {
                            text: qsTr("Transformer Tank")
                        }
                        CheckBox {
                            text: qsTr("HV Bushing")
                        }
                        CheckBox {
                            text: qsTr("LV Bushing")
                        }
                    }
                    
                    
                }
            }
        }
        
        
    }

    
}