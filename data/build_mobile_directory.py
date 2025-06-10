#!/usr/bin/env python3
"""
Script pour ajouter la structure mobile à YOLO-One
À exécuter APRÈS build_directory.py
"""

from pathlib import Path

def create_mobile_structure():
    """Crée la structure mobile complète"""
    
    print("📱 Création de la structure mobile YOLO-One...")
    
    # Structure mobile complète
    mobile_dirs_and_files = {
        "mobile": ["README.md", "build_mobile.sh"],
        
        # Android
        "mobile/android": ["build.gradle", "settings.gradle"],
        "mobile/android/app": ["build.gradle"],
        "mobile/android/app/src/main/java/com/iatrax/yoloone": [
            "MainActivity.java", 
            "YOLOOneDetector.java"
        ],
        "mobile/android/app/src/main/cpp": [
            "native-lib.cpp"
        ],
        "mobile/android/app/src/main/assets": [
            "models.txt"
        ],
        
        # iOS
        "mobile/ios": ["Podfile", "README.md"],
        "mobile/ios/YOLOOne": [
            "ContentView.swift",
            "AppDelegate.swift"
        ],
        "mobile/ios/YOLOOne/Models": [
            "YOLOOneDetector.swift"
        ],
        
        # Shared
        "mobile/shared": [
            "model_converter.py",
            "benchmark_mobile.py"
        ]
    }
    
    # Créer tout
    for folder, files in mobile_dirs_and_files.items():
        # Créer le dossier
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"📂 Dossier créé: {folder}/")
        
        # Créer les fichiers
        for file in files:
            file_path = Path(folder) / file
            
            # Contenu basique selon le type
            if file.endswith('.java'):
                content = f"// {file}\n// TODO: Implémenter classe Java\npublic class {file.replace('.java', '')} {{\n    // Code ici\n}}"
            elif file.endswith('.swift'):
                content = f"// {file}\n// TODO: Implémenter SwiftUI\nimport SwiftUI\n\nstruct {file.replace('.swift', '')}: View {{\n    var body: some View {{\n        Text(\"Hello YOLO-One\")\n    }}\n}}"
            elif file.endswith('.py'):
                content = f"#!/usr/bin/env python3\n\"\"\"\n{file} - YOLO-One Mobile\n\"\"\"\n\n# TODO: Implémenter\nprint('YOLO-One Mobile Tools')"
            else:
                content = f"# {file}\n# TODO: Configuration pour {file}"
            
            file_path.write_text(content, encoding='utf-8')
            print(f"📄 Fichier créé: {folder}/{file}")
    
    print("\n✅ Structure mobile créée avec succès !")
    print("\n📱 Dossiers créés :")
    print("   🤖 mobile/android/     - App Android")
    print("   🍎 mobile/ios/         - App iOS") 
    print("   🔧 mobile/shared/      - Outils partagés")

if __name__ == "__main__":
    create_mobile_structure()